from diffusers import DDIMScheduler, StableDiffusionPipeline
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from peft import LoraConfig


class StableDiffusion(nn.Module):
    def __init__(self, args, t_range=[0.02, 0.98]):
        super().__init__()

        self.device = args.device
        self.dtype = args.precision
        print(f'[INFO] Loading Stable Diffusion...')

        model_key = "sd2-community/stable-diffusion-2-1-base"
        pipe = StableDiffusionPipeline.from_pretrained(
            model_key, torch_dtype=self.dtype,
        )

        pipe.to(self.device)
        self.vae = pipe.vae.eval()
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet.eval()
        
        # Freeze models
        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)
        
        # Schedulers
        self.scheduler = DDIMScheduler.from_pretrained(
            model_key, subfolder="scheduler", torch_dtype=self.dtype,
        )
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)

        del pipe

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod
        
        print(f'[INFO] Loaded Stable Diffusion!')
        
        # Initialize VSD components if needed
        if args.loss_type == "vsd":
            self._init_vsd_components(args.lora_rank)
    
    def _init_vsd_components(self, lora_rank=4):
        """Initialize LoRA for VSD"""
        print(f"[INFO] Initializing VSD with LoRA rank={lora_rank}")
        
        self.unet.requires_grad_(False)
        
        unet_lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        
        self.unet.add_adapter(unet_lora_config)
        self.lora_layers = list(filter(lambda p: p.requires_grad, self.unet.parameters()))

    @torch.no_grad()
    def get_text_embeds(self, prompt):
        """Get text embeddings from prompt"""
        inputs = self.tokenizer(
            prompt, 
            padding='max_length', 
            max_length=self.tokenizer.model_max_length, 
            return_tensors='pt'
        )
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]
        return embeddings
    
    def get_noise_preds(self, latents_noisy, t, text_embeddings, guidance_scale=7.5):
        """
        Predict noise with Classifier-Free Guidance (CFG)
        
        CFG formula: noise_pred = uncond + guidance_scale * (cond - uncond)
        """
        latent_model_input = torch.cat([latents_noisy] * 2)
        tt = torch.cat([t] * 2)
        
        noise_pred = self.unet(latent_model_input, tt, encoder_hidden_states=text_embeddings).sample
        noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
        
        # Classifier-Free Guidance
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_pos - noise_pred_uncond)
        
        return noise_pred
    
    def get_sds_loss(self, latents, text_embeddings, guidance_scale=7.5):
        """
        Score Distillation Sampling (SDS) loss.

        Reference: DreamFusion (https://arxiv.org/abs/2209.14988)

        Adds random noise to the latents at a random timestep and lets the
        diffusion model predict it. The difference between predicted and added
        noise is used as the gradient on the latents.

        Args:
            latents: Current optimized latents (B, 4, H, W)
            text_embeddings: Concatenated [uncond, cond] embeddings (2*B, seq_len, dim)
            guidance_scale: CFG scale

        Returns:
            loss: Scalar SDS loss
        """
        # TODO: Implement SDS loss
        B = latents.shape[0]  # 取得 batch size，也就是一次處理幾筆資料
        # 隨機選擇 diffusion timestep
        t = torch.randint(
            self.min_step, #可抽取的最小 timestep
            self.max_step + 1,
            (B,),
            device=latents.device,
            dtype=torch.long
        )
        
        # 產生 Gaussian noise
        # 產生與 latents 形狀相同的隨機數值
        noise = torch.randn_like(latents)
        
        # 取得 alpha_t
        alpha = self.alphas[t].to(
            device=latents.device,
            dtype=latents.dtype
        )
        alpha = alpha.view(B, 1, 1, 1)
        
        # 使用 scheduler 內建的函式，將 noise 加到 latents
        # 根據 scheduler 內部的累積 alpha 值，計算對應 timestep 的加噪結果
        latents_noisy = self.scheduler.add_noise(
            latents,
            noise,
            t
        )
        
        # 使用 classifier-free guidance 預測 noise
        noise_pred = self.get_noise_preds(
            latents_noisy,
            t,
            text_embeddings,
            guidance_scale=guidance_scale
        )
        
        # 計算 SDS gradient
        # (1.0 - alpha)是與 timestep 有關的權重 會影響模型預測誤差對梯度的貢獻
        grad = (1.0 - alpha) * (noise_pred - noise)
        grad = torch.nan_to_num(grad)
        
        # 將 gradient 轉換成 surrogate MSE loss
        # 為了讓 loss 的反向傳播產生希望的梯度方向而建立的
        target = (latents - grad).detach()
        
        loss = 0.5 * F.mse_loss(
            latents,
            target,
            reduction="sum"
        ) / B

        return loss
        #raise NotImplementedError("TODO: Implement SDS loss")

    def get_vsd_loss(self, latents, text_embeddings, guidance_scale=7.5, lora_loss_weight=1.0):
        """
        Variational Score Distillation (VSD) loss.

        Reference: ProlificDreamer (https://arxiv.org/abs/2305.16213)

        Same as SDS, but the added noise is replaced by the prediction of a
        LoRA-finetuned UNet that models the distribution of the current latents.
        The LoRA is trained jointly with a standard denoising loss.

        Args:
            lora_loss_weight: Weight of the LoRA training loss

        Returns:
            loss: Scalar VSD loss + lora_loss_weight * LoRA loss
        """
        # TODO: Implement VSD loss
        raise NotImplementedError("TODO: Implement VSD loss")

    @torch.no_grad()
    def invert_noise(self, latents, target_t, text_embeddings, guidance_scale=-7.5, n_steps=10, eta=0.3):
        """
        DDIM inversion: x0 -> x_t.

        Reference: Score Distillation via Reparametrized DDIM (https://arxiv.org/abs/2405.15891)

        Runs the DDIM update in reverse, from t=0 up to `target_t`, to find noisy
        latents consistent with x0. Write the loop yourself: you must NOT call any
        "invert" / "ddim_invert" / "invert_step" utilities from diffusers or other libraries.

        Args:
            target_t: Target timestep to invert to
            guidance_scale: CFG scale (typically negative for inversion)
            n_steps: Number of inversion steps
            eta: Stochasticity level (0 = deterministic)

        Returns:
            latents_noisy: Inverted noisy latents x_t (B, 4, H, W)
        """
        # TODO: Implement DDIM inversion
        # Hint: You may assume eta=0 (deterministic inversion) and ignore `eta`;
        # no points will be deducted for this.
        raise NotImplementedError("TODO: Implement DDIM inversion")

    def get_sdi_loss(
        self,
        latents,
        text_embeddings,
        guidance_scale=7.5,
        current_iter=0,
        total_iters=500,
        inversion_guidance_scale=-7.5,
        inversion_n_steps=10,
        inversion_eta=0.3,
        update_interval=25,
    ):
        """
        Score Distillation via Inversion (SDI) loss.

        Reference: Score Distillation via Reparametrized DDIM (https://arxiv.org/abs/2405.15891)

        Instead of random noise as in SDS, SDI gets the noise by DDIM inversion of
        the current latents. The timestep is annealed from max_step to min_step, and
        the denoised target is cached and refreshed every `update_interval` iterations.

        Args:
            current_iter: Current training iteration
            total_iters: Total number of training iterations
            inversion_guidance_scale, inversion_n_steps, inversion_eta:
                Passed to `invert_noise` as guidance_scale, n_steps, eta
            update_interval: Refresh the cached target every N iterations

        Returns:
            loss: Scalar MSE loss between current latents and the cached target
        """
        B = latents.shape[0]

        # TODO: Compute the annealed timestep for the current iteration
        t = None

        # Check if we need to update target
        should_update = (current_iter % update_interval == 0) or not hasattr(self, 'sdi_target')

        if should_update:
            with torch.no_grad():
                # Perform DDIM inversion: x0 -> x_t
                latents_noisy = self.invert_noise(
                    latents, t, text_embeddings,
                    guidance_scale=inversion_guidance_scale,
                    n_steps=inversion_n_steps,
                    eta=inversion_eta
                )

                # TODO: Predict noise from the inverted noisy latents
                noise_pred = None

                # TODO: Denoise to get the target x0 from the predicted noise
                target = None

                # Cache the target
                self.sdi_target = target.detach()

        # TODO: Compute the MSE loss between current latents and the cached target
        loss = None

        return loss
        
    @torch.no_grad()
    def decode_latents(self, latents):
        """Decode latents to RGB images"""
        latents = 1 / self.vae.config.scaling_factor * latents
        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs
