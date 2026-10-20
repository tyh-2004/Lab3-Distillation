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
        # ---------------------------------------------------------
        # 0. 確認 LoRA 已經初始化
        # ---------------------------------------------------------
        # VSD 需要使用 LoRA UNet 來估計目前 latent 分布的 noise。
        # 因此必須先在 __init__() 中以 loss_type="vsd" 初始化 LoRA。
        if not hasattr(self, "lora_layers"):
            raise RuntimeError(
                "VSD requires LoRA components. "
                "Please initialize the model with loss_type='vsd'."
            )

        # 取得 batch size
        B = latents.shape[0]

        # ---------------------------------------------------------
        # 1. 隨機選擇 diffusion timestep
        # ---------------------------------------------------------
        #
        # 每個 batch sample 都可以使用不同的 timestep。
        # timestep 限制在 [min_step, max_step]，
        # 避免太接近乾淨影像或純噪聲的區域。
        #
        t = torch.randint(
            self.min_step,
            self.max_step + 1,
            (B,),
            device=latents.device,
            dtype=torch.long,
        )

        # ---------------------------------------------------------
        # 2. 產生 Gaussian noise
        # ---------------------------------------------------------
        #
        # noise 是稍後加入目前 latents 的真實噪聲。
        #
        noise = torch.randn_like(latents)

        # ---------------------------------------------------------
        # 3. 使用 scheduler 對 latents 加噪
        # ---------------------------------------------------------
        #
        # scheduler.add_noise() 會依據正確的 alpha_bar_t 執行：
        #
        #     x_t = sqrt(alpha_bar_t) * x_0
        #           + sqrt(1 - alpha_bar_t) * noise
        #
        # 使用 scheduler 內建函式，可以避免錯把單步 alpha_t
        # 當成累積乘積 alpha_bar_t。
        #
        latents_noisy = self.scheduler.add_noise(
            latents,
            noise,
            t,
        )
        # 給 LoRA UNet 使用的 noisy latents
        # 使用 latents.detach()，避免 lora_loss 更新原始 latents
        latents_noisy_lora = self.scheduler.add_noise(
            latents.detach(),
            noise,
            t,
        )
        # ---------------------------------------------------------
        # 4. 使用「原始 UNet」預測 noise
        # ---------------------------------------------------------
        # VSD 需要兩個 noise prediction：
        #
        # 1. 原始 Stable Diffusion UNet：
        #        noise_pred_base
        #
        # 2. 加上 LoRA 後的 UNet：
        #        noise_pred_lora
        #
        # 原始 UNet 的參數是 frozen 的，
        # 所以這裡不需要建立 gradient graph。
        #
        # disable_adapters() 會暫時關閉 LoRA，
        # 讓 UNet 使用原本的 Stable Diffusion 權重。
        #
        self.unet.disable_adapters()

        try:
            with torch.no_grad():
                noise_pred_base = self.get_noise_preds(
                    latents_noisy,
                    t,
                    text_embeddings,
                    guidance_scale=guidance_scale,
                )
        finally:
            # 不管前面的 forward 是否發生錯誤，
            # 都要重新開啟 LoRA，避免影響後續訓練。
            self.unet.enable_adapters()

        # ---------------------------------------------------------
        # 5. 使用「LoRA UNet」預測 noise
        # ---------------------------------------------------------
        #
        # 這次保留 gradient graph，因為 LoRA 參數需要透過
        # lora_loss 進行更新。
        #
        noise_pred_lora = self.get_noise_preds(
            latents_noisy_lora,
            t,
            text_embeddings,
            guidance_scale=guidance_scale
            # guidance_scale=1.0
        )

        # ---------------------------------------------------------
        # 6. 計算 VSD 的 timestep weighting
        # ---------------------------------------------------------
        # VSD 和 SDS 一樣，通常使用：
        #     w(t) = 1 - alpha_bar_t
        # 這裡直接使用 scheduler 的 alphas_cumprod，
        # 確保取得的是累積乘積 alpha_bar_t。
        # 修過
        alphas_cumprod = self.scheduler.alphas_cumprod.to(
            device=latents.device,
            dtype=latents.dtype,
        )

        # 將 (B,) reshape 為 (B, 1, 1, 1)，
        # 以便和 (B, 4, H, W) 的 tensor broadcasting。
        alpha_bar = alphas_cumprod[t].view(B, 1, 1, 1)

        # ---------------------------------------------------------
        # 7. 計算 VSD distillation gradient
        # ---------------------------------------------------------
        #
        # VSD 的核心和 SDS 不同：
        #
        # SDS：
        #     grad = w(t) * (noise_pred_base - noise)
        #
        # VSD：
        #     grad = w(t) * (noise_pred_base - noise_pred_lora)
        #
        # 原始 UNet 提供 pretrained diffusion distribution，
        # LoRA UNet 則學習目前正在優化的 latent 分布。
        #
        # detach noise_pred_lora 的原因：
        # distillation loss 主要用來更新 latents，
        # LoRA 本身則透過下面的 lora_loss 更新。
        #
        grad = (1.0 - alpha_bar) * (
            noise_pred_base - noise_pred_lora.detach()
        )

        # 避免數值不穩定造成 NaN 或 Inf。
        grad = torch.nan_to_num(grad)

        # ---------------------------------------------------------
        # 8. 將 VSD gradient 轉換成 surrogate loss
        # ---------------------------------------------------------
        #
        # 透過 detached target，讓一般的 MSE loss
        # 對 latents 產生 VSD 想要的更新方向。
        #
        target = (latents - grad).detach()

        distillation_loss = 0.5 * F.mse_loss(
            latents,
            target,
            reduction="sum",
        ) / B

        # ---------------------------------------------------------
        # 9. 計算 LoRA denoising loss
        # ---------------------------------------------------------
        #
        # LoRA UNet 需要學會預測實際加入的 noise。
        #
        # 這個 loss 會保留對 LoRA 參數的 gradient，
        # 因此可以用來訓練 self.lora_layers。
        #
        lora_loss = F.mse_loss(
            noise_pred_lora,
            noise,
            reduction="mean"
        )

        # ---------------------------------------------------------
        # 10. 合併 VSD loss 和 LoRA training loss
        # ---------------------------------------------------------
        # 最終 loss
        loss = (distillation_loss + lora_loss_weight * lora_loss)

        return loss
        #raise NotImplementedError("TODO: Implement VSD loss")

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
        # ---------------------------------------------------------
        # 1. 準備基本參數
        # ---------------------------------------------------------
        # 取得 batch size
        B = latents.shape[0]
        # target_t 可能是：
        # - 一個 scalar tensor
        # - 形狀為 (B,) 的 tensor
        # - Python integer
        # 這裡假設同一個 batch 中的 sample 使用相同 target timestep，
        # 因此取第一個 timestep 作為整個 batch 的目標。
        if torch.is_tensor(target_t):
            target_t_value = int(target_t.flatten()[0].item())
        else:
            target_t_value = int(target_t)
        # 確保 target timestep 在合法範圍內
        target_t_value = max(
            0,
            min(target_t_value, self.num_train_timesteps - 1),
        )

        # 如果目標 timestep 是 0，代表不需要加噪，
        # 目前的 latents 就是結果。
        if target_t_value == 0:
            return latents
        # 確保 n_steps 至少為 1
        n_steps = max(1, int(n_steps))
        # ---------------------------------------------------------
        # 2. 建立反演過程中的 timestep 序列
        # ---------------------------------------------------------
        # DDIM inversion 要從低噪聲 timestep 逐步走向高噪聲 timestep：
        #     t_0 -> t_1 -> ... -> target_t
        # round() 可以避免 linspace 產生浮點數後直接轉 int
        # 所造成的截斷誤差。
        inversion_timesteps = torch.linspace(
            0,
            target_t_value,
            n_steps + 1,
            device=latents.device,
            dtype=torch.float32,
        ).round().long()
        # 目前的 latent 一開始就是乾淨 latent x_0
        latents_noisy = latents
        # 直接使用 scheduler 中的 cumulative alpha：
        #     alpha_bar_t = self.scheduler.alphas_cumprod[t]
        # 這裡不能使用單步 alpha_t。
        alphas_cumprod = self.scheduler.alphas_cumprod.to(
            device=latents.device,
            dtype=latents.dtype,
        )
        # ---------------------------------------------------------
        # 3. 逐步執行 DDIM inversion
        # ---------------------------------------------------------
        # 每一步會從 x_t 推進到 x_next：
        #
        #     x_t -> x_next
        #
        # 這是 deterministic DDIM inversion，因此忽略 eta，
        # 不另外加入隨機噪聲。
        #
        for step_index in range(len(inversion_timesteps) - 1):

            # 目前 timestep 與下一個 timestep
            current_t = inversion_timesteps[step_index]
            next_t = inversion_timesteps[step_index + 1]

            # UNet 需要形狀為 (B,) 的 timestep tensor
            current_t_batch = current_t.expand(B)

            # -----------------------------------------------------
            # 3a. 預測目前 timestep 的 noise
            # -----------------------------------------------------
            # guidance_scale 通常使用負值進行 inversion，
            # 這是 SDI 作業設定中的反向 guidance 做法。
            #
            noise_pred = self.get_noise_preds(
                latents_noisy,
                current_t_batch,
                text_embeddings,
                guidance_scale=guidance_scale,
            )

            # -----------------------------------------------------
            # 3b. 取得 current_t 和 next_t 的 alpha_bar
            # -----------------------------------------------------
            #
            # alpha_bar_current 對每個 batch sample 都相同，
            # 因此 shape 為 (B, 1, 1, 1)。
            #
            alpha_bar_current = alphas_cumprod[current_t].view(
                1, 1, 1, 1
            )

            # next_t 是單一 timestep，因此可以使用 scalar alpha，
            # shape 為 (1, 1, 1, 1)，會自動 broadcast 到 batch。
            alpha_bar_next = alphas_cumprod[next_t].view(
                1, 1, 1, 1
            )

            # 避免 sqrt(0) 或數值誤差造成不穩定。
            alpha_bar_current = alpha_bar_current.clamp(min=1e-5, max=1.0)
            alpha_bar_next = alpha_bar_next.clamp(min=1e-5, max=1.0)

            # -----------------------------------------------------
            # 3c. 根據目前的 noise prediction 估計 x_0
            # -----------------------------------------------------
            #
            # 正向 diffusion 公式為：
            #
            #     x_t = sqrt(alpha_bar_t) * x_0
            #           + sqrt(1 - alpha_bar_t) * noise
            #
            # 移項後可以估計乾淨 latent：
            #
            #     x_0_pred =
            #         (x_t - sqrt(1-alpha_bar_t)*noise_pred)
            #         / sqrt(alpha_bar_t)
            #
            x0_pred = (
                latents_noisy
                - torch.sqrt(1.0 - alpha_bar_current) * noise_pred
            ) / torch.sqrt(alpha_bar_current)

            # -----------------------------------------------------
            # 3d. 使用 x_0_pred 和 noise_pred 推進到 next_t
            # -----------------------------------------------------
            #
            # 將估計出的 x_0_pred 放入 next timestep 的
            # diffusion equation：
            #
            #     x_next =
            #         sqrt(alpha_bar_next) * x_0_pred
            #         + sqrt(1-alpha_bar_next) * noise_pred
            #
            # 這就是 deterministic DDIM inversion 的核心更新。
            #
            latents_noisy = (
                torch.sqrt(alpha_bar_next) * x0_pred
                + torch.sqrt(1.0 - alpha_bar_next) * noise_pred
            )

        # 回傳 target_t 對應的 noisy latent x_target_t
        return latents_noisy
        #raise NotImplementedError("TODO: Implement DDIM inversion")

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
