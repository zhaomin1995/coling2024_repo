# flan_t5 + target
CUDA_VISIBLE_DEVICES=0 python llm.py -data_dir data/ -experiment flan_t5 -input_content target -output_dir outputs/flan -cache_dir /mnt/DATA/hf_cache/ -exemplar zero-shot &
CUDA_VISIBLE_DEVICES=1 python llm.py -data_dir data/ -experiment flan_t5 -input_content target -output_dir outputs/flan -cache_dir /mnt/DATA/hf_cache/ -exemplar one-shot &
CUDA_VISIBLE_DEVICES=2 python llm.py -data_dir data/ -experiment flan_t5 -input_content target -output_dir outputs/flan -cache_dir /mnt/DATA/hf_cache/ -exemplar five-shot &
CUDA_VISIBLE_DEVICES=3 python llm.py -data_dir data/ -experiment flan_t5 -input_content target -output_dir outputs/flan -cache_dir /mnt/DATA/hf_cache/ -exemplar ten-shot

# flan_t5 + early_target
CUDA_VISIBLE_DEVICES=0 python llm.py -data_dir data/ -experiment flan_t5 -input_content early_target -output_dir outputs/flan -cache_dir /mnt/DATA/hf_cache/ -exemplar zero-shot &
CUDA_VISIBLE_DEVICES=1 python llm.py -data_dir data/ -experiment flan_t5 -input_content early_target -output_dir outputs/flan -cache_dir /mnt/DATA/hf_cache/ -exemplar one-shot &
CUDA_VISIBLE_DEVICES=2 python llm.py -data_dir data/ -experiment flan_t5 -input_content early_target -output_dir outputs/flan -cache_dir /mnt/DATA/hf_cache/ -exemplar five-shot &
CUDA_VISIBLE_DEVICES=3 python llm.py -data_dir data/ -experiment flan_t5 -input_content early_target -output_dir outputs/flan -cache_dir /mnt/DATA/hf_cache/ -exemplar ten-shot

# flan_t5 + target_later
CUDA_VISIBLE_DEVICES=0 python llm.py -data_dir data/ -experiment flan_t5 -input_content target_later -output_dir outputs/flan -cache_dir /mnt/DATA/hf_cache/ -exemplar zero-shot &
CUDA_VISIBLE_DEVICES=1 python llm.py -data_dir data/ -experiment flan_t5 -input_content target_later -output_dir outputs/flan -cache_dir /mnt/DATA/hf_cache/ -exemplar one-shot &
CUDA_VISIBLE_DEVICES=2 python llm.py -data_dir data/ -experiment flan_t5 -input_content target_later -output_dir outputs/flan -cache_dir /mnt/DATA/hf_cache/ -exemplar five-shot &
CUDA_VISIBLE_DEVICES=3 python llm.py -data_dir data/ -experiment flan_t5 -input_content target_later -output_dir outputs/flan -cache_dir /mnt/DATA/hf_cache/ -exemplar ten-shot

# flan_t5 + all
CUDA_VISIBLE_DEVICES=0 python llm.py -data_dir data/ -experiment flan_t5 -input_content all -output_dir outputs/flan -cache_dir /mnt/DATA/hf_cache/ -exemplar zero-shot &
CUDA_VISIBLE_DEVICES=1 python llm.py -data_dir data/ -experiment flan_t5 -input_content all -output_dir outputs/flan -cache_dir /mnt/DATA/hf_cache/ -exemplar one-shot &
CUDA_VISIBLE_DEVICES=2 python llm.py -data_dir data/ -experiment flan_t5 -input_content all -output_dir outputs/flan -cache_dir /mnt/DATA/hf_cache/ -exemplar five-shot &
CUDA_VISIBLE_DEVICES=3 python llm.py -data_dir data/ -experiment flan_t5 -input_content all -output_dir outputs/flan -cache_dir /mnt/DATA/hf_cache/ -exemplar ten-shot


# flan_ul2 + target
CUDA_VISIBLE_DEVICES=0 python llm.py -data_dir data/ -experiment flan_ul2 -input_content target -output_dir outputs/flan -cache_dir /mnt/DATA/hf_cache/ -exemplar zero-shot &
CUDA_VISIBLE_DEVICES=1 python llm.py -data_dir data/ -experiment flan_ul2 -input_content target -output_dir outputs/flan -cache_dir /mnt/DATA/hf_cache/ -exemplar one-shot &
CUDA_VISIBLE_DEVICES=2 python llm.py -data_dir data/ -experiment flan_ul2 -input_content target -output_dir outputs/flan -cache_dir /mnt/DATA/hf_cache/ -exemplar five-shot &
CUDA_VISIBLE_DEVICES=3 python llm.py -data_dir data/ -experiment flan_ul2 -input_content target -output_dir outputs/flan -cache_dir /mnt/DATA/hf_cache/ -exemplar ten-shot

# flan_ul2 + early_target
CUDA_VISIBLE_DEVICES=0 python llm.py -data_dir data/ -experiment flan_ul2 -input_content early_target -output_dir outputs/flan -cache_dir /mnt/DATA/hf_cache/ -exemplar zero-shot &
CUDA_VISIBLE_DEVICES=1 python llm.py -data_dir data/ -experiment flan_ul2 -input_content early_target -output_dir outputs/flan -cache_dir /mnt/DATA/hf_cache/ -exemplar one-shot &
CUDA_VISIBLE_DEVICES=2 python llm.py -data_dir data/ -experiment flan_ul2 -input_content early_target -output_dir outputs/flan -cache_dir /mnt/DATA/hf_cache/ -exemplar five-shot &
CUDA_VISIBLE_DEVICES=3 python llm.py -data_dir data/ -experiment flan_ul2 -input_content early_target -output_dir outputs/flan -cache_dir /mnt/DATA/hf_cache/ -exemplar ten-shot

# flan_ul2 + target_later
CUDA_VISIBLE_DEVICES=0 python llm.py -data_dir data/ -experiment flan_ul2 -input_content target_later -output_dir outputs/flan -cache_dir /mnt/DATA/hf_cache/ -exemplar zero-shot &
CUDA_VISIBLE_DEVICES=1 python llm.py -data_dir data/ -experiment flan_ul2 -input_content target_later -output_dir outputs/flan -cache_dir /mnt/DATA/hf_cache/ -exemplar one-shot &
CUDA_VISIBLE_DEVICES=2 python llm.py -data_dir data/ -experiment flan_ul2 -input_content target_later -output_dir outputs/flan -cache_dir /mnt/DATA/hf_cache/ -exemplar five-shot &
CUDA_VISIBLE_DEVICES=3 python llm.py -data_dir data/ -experiment flan_ul2 -input_content target_later -output_dir outputs/flan -cache_dir /mnt/DATA/hf_cache/ -exemplar ten-shot

# flan_ul2 + all
CUDA_VISIBLE_DEVICES=0 python llm.py -data_dir data/ -experiment flan_ul2 -input_content all -output_dir outputs/flan -cache_dir /mnt/DATA/hf_cache/ -exemplar zero-shot &
CUDA_VISIBLE_DEVICES=1 python llm.py -data_dir data/ -experiment flan_ul2 -input_content all -output_dir outputs/flan -cache_dir /mnt/DATA/hf_cache/ -exemplar one-shot &
CUDA_VISIBLE_DEVICES=2 python llm.py -data_dir data/ -experiment flan_ul2 -input_content all -output_dir outputs/flan -cache_dir /mnt/DATA/hf_cache/ -exemplar five-shot &
CUDA_VISIBLE_DEVICES=3 python llm.py -data_dir data/ -experiment flan_ul2 -input_content all -output_dir outputs/flan -cache_dir /mnt/DATA/hf_cache/ -exemplar ten-shot


# flan_alpaca + target
CUDA_VISIBLE_DEVICES=0 python llm.py -data_dir data/ -experiment flan_alpaca -input_content target -output_dir outputs/flan -cache_dir /mnt/DATA/hf_cache/ -exemplar zero-shot &
CUDA_VISIBLE_DEVICES=1 python llm.py -data_dir data/ -experiment flan_alpaca -input_content target -output_dir outputs/flan -cache_dir /mnt/DATA/hf_cache/ -exemplar one-shot &
CUDA_VISIBLE_DEVICES=2 python llm.py -data_dir data/ -experiment flan_alpaca -input_content target -output_dir outputs/flan -cache_dir /mnt/DATA/hf_cache/ -exemplar five-shot &
CUDA_VISIBLE_DEVICES=3 python llm.py -data_dir data/ -experiment flan_alpaca -input_content target -output_dir outputs/flan -cache_dir /mnt/DATA/hf_cache/ -exemplar ten-shot

# flan_alpaca + early_target
CUDA_VISIBLE_DEVICES=0 python llm.py -data_dir data/ -experiment flan_alpaca -input_content early_target -output_dir outputs/flan -cache_dir /mnt/DATA/hf_cache/ -exemplar zero-shot &
CUDA_VISIBLE_DEVICES=1 python llm.py -data_dir data/ -experiment flan_alpaca -input_content early_target -output_dir outputs/flan -cache_dir /mnt/DATA/hf_cache/ -exemplar one-shot &
CUDA_VISIBLE_DEVICES=2 python llm.py -data_dir data/ -experiment flan_alpaca -input_content early_target -output_dir outputs/flan -cache_dir /mnt/DATA/hf_cache/ -exemplar five-shot &
CUDA_VISIBLE_DEVICES=3 python llm.py -data_dir data/ -experiment flan_alpaca -input_content early_target -output_dir outputs/flan -cache_dir /mnt/DATA/hf_cache/ -exemplar ten-shot

# flan_alpaca + target_later
CUDA_VISIBLE_DEVICES=0 python llm.py -data_dir data/ -experiment flan_alpaca -input_content target_later -output_dir outputs/flan -cache_dir /mnt/DATA/hf_cache/ -exemplar zero-shot &
CUDA_VISIBLE_DEVICES=1 python llm.py -data_dir data/ -experiment flan_alpaca -input_content target_later -output_dir outputs/flan -cache_dir /mnt/DATA/hf_cache/ -exemplar one-shot &
CUDA_VISIBLE_DEVICES=2 python llm.py -data_dir data/ -experiment flan_alpaca -input_content target_later -output_dir outputs/flan -cache_dir /mnt/DATA/hf_cache/ -exemplar five-shot &
CUDA_VISIBLE_DEVICES=3 python llm.py -data_dir data/ -experiment flan_alpaca -input_content target_later -output_dir outputs/flan -cache_dir /mnt/DATA/hf_cache/ -exemplar ten-shot

# flan_alpaca + all
CUDA_VISIBLE_DEVICES=0 python llm.py -data_dir data/ -experiment flan_alpaca -input_content all -output_dir outputs/flan -cache_dir /mnt/DATA/hf_cache/ -exemplar zero-shot &
CUDA_VISIBLE_DEVICES=1 python llm.py -data_dir data/ -experiment flan_alpaca -input_content all -output_dir outputs/flan -cache_dir /mnt/DATA/hf_cache/ -exemplar one-shot &
CUDA_VISIBLE_DEVICES=2 python llm.py -data_dir data/ -experiment flan_alpaca -input_content all -output_dir outputs/flan -cache_dir /mnt/DATA/hf_cache/ -exemplar five-shot &
CUDA_VISIBLE_DEVICES=3 python llm.py -data_dir data/ -experiment flan_alpaca -input_content all -output_dir outputs/flan -cache_dir /mnt/DATA/hf_cache/ -exemplar ten-shot
