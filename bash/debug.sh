

CUDA_VISIBLE_DEVICES=0 python eval_dd.py --input outputs_final/common_person_bench/ip-adapter_faceid/

CUDA_VISIBLE_DEVICES=1 python eval_dd.py --input outputs_final/common_person_bench/ip-adapter_sd15/

CUDA_VISIBLE_DEVICES=2 python eval_dd.py --input outputs_final/common_person_bench/ip-adapter-plus_sd15

CUDA_VISIBLE_DEVICES=2 python eval_dd.py --input outputs_final/common_person_bench/sd15_latent_new_lr_1e-5_4a100-motion-update-1112-with-ref-noisy-cross-attn-whitebg-head-frame_stride-stride-4-8card/checkpoint-100000

CUDA_VISIBLE_DEVICES=3 python eval_dd.py --input outputs_final/common_person_bench/sd15_latent_new_lr_1e-5_4a100-motion-update-1112-with-ref-noisy-cross-attn-whitebg-head-frame_stride-stride-4-8card/checkpoint-150000

CUDA_VISIBLE_DEVICES=4 python eval_dd.py --input outputs_final/common_person_bench/photomaker

CUDA_VISIBLE_DEVICES=5 python eval_dd.py --input outputs_final/common_person_bench/sd15_latent_new_lr_1e-5_4a100-motion-update-1107-with-ref-noisy-cross-attn-whitebg-head-frame_stride-stride-8-4card/checkpoint-180000

CUDA_VISIBLE_DEVICES=6 python eval_dd.py --input outputs_final/common_person_bench/sd15_latent_new_lr_1e-5_4a100-motion-update-1107-with-ref-noisy-cross-attn-whitebg-head-frame_stride-stride-8-4card/checkpoint-200000

CUDA_VISIBLE_DEVICES=7 python eval_dd.py --input /group/40034/jackeywu/code/ID-Animator/outputs/common_bench

