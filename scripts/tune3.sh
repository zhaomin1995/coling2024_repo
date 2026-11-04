# flan_t5 + all
CUDA_VISIBLE_DEVICES=3 python llm.py -data_dir data/ -experiment flan_t5+ -input_content all -output_dir outputs/flan -cache_dir /mnt/DATA/hf_cache/ -exemplar zero-shot
CUDA_VISIBLE_DEVICES=3 python llm.py -data_dir data/ -experiment flan_t5+ -input_content all -output_dir outputs/flan -cache_dir /mnt/DATA/hf_cache/ -exemplar one-shot
CUDA_VISIBLE_DEVICES=3 python llm.py -data_dir data/ -experiment flan_t5+ -input_content all -output_dir outputs/flan -cache_dir /mnt/DATA/hf_cache/ -exemplar five-shot
CUDA_VISIBLE_DEVICES=3 python llm.py -data_dir data/ -experiment flan_t5+ -input_content all -output_dir outputs/flan -cache_dir /mnt/DATA/hf_cache/ -exemplar ten-shot

# flan_ul2 + all
CUDA_VISIBLE_DEVICES=3 python llm.py -data_dir data/ -experiment flan_ul2+ -input_content all -output_dir outputs/flan -cache_dir /mnt/DATA/hf_cache/ -exemplar zero-shot
CUDA_VISIBLE_DEVICES=3 python llm.py -data_dir data/ -experiment flan_ul2+ -input_content all -output_dir outputs/flan -cache_dir /mnt/DATA/hf_cache/ -exemplar one-shot
CUDA_VISIBLE_DEVICES=3 python llm.py -data_dir data/ -experiment flan_ul2+ -input_content all -output_dir outputs/flan -cache_dir /mnt/DATA/hf_cache/ -exemplar five-shot
CUDA_VISIBLE_DEVICES=3 python llm.py -data_dir data/ -experiment flan_ul2+ -input_content all -output_dir outputs/flan -cache_dir /mnt/DATA/hf_cache/ -exemplar ten-shot

# flan_alpaca + all
CUDA_VISIBLE_DEVICES=3 python llm.py -data_dir data/ -experiment flan_alpaca+ -input_content all -output_dir outputs/flan -cache_dir /mnt/DATA/hf_cache/ -exemplar zero-shot
CUDA_VISIBLE_DEVICES=3 python llm.py -data_dir data/ -experiment flan_alpaca+ -input_content all -output_dir outputs/flan -cache_dir /mnt/DATA/hf_cache/ -exemplar one-shot
CUDA_VISIBLE_DEVICES=3 python llm.py -data_dir data/ -experiment flan_alpaca+ -input_content all -output_dir outputs/flan -cache_dir /mnt/DATA/hf_cache/ -exemplar five-shot
CUDA_VISIBLE_DEVICES=3 python llm.py -data_dir data/ -experiment flan_alpaca+ -input_content all -output_dir outputs/flan -cache_dir /mnt/DATA/hf_cache/ -exemplar ten-shot