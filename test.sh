# python test_animate.py \
#     -i "datasets/characters/real/Harry_Potter/image" \
#     -o "photomaker_harry_potter" \
#     --name "Harry_Potter"

# python test_animate.py \
#     -i "datasets/characters/real/Harry_Potter/image" \
#     -o "photomakeradapter_clipG_harry_potter" \
#     --name "Harry_Potter" \
#     --ip_adapter

# python test_animate.py \
#     -i "datasets/characters/real/Harry_Potter/image" \
#     -o "photomakeradapter_harry_potter" \
#     --name "Harry_Potter" \
#     --ip_adapter \
#     --use_clipl_embed
    
# python test_animate.py \
#     -i "datasets/benchmark_dataset/person_1" \
#     --name "person_1" \
#     -o "photomaker_person_1"

# python test_animate.py \
#     -i "datasets/benchmark_dataset/person_1" \
#     --name "person_1" \
#     -o "photomakeradapter_clipG_person_1" \
#     --ip_adapter

# python test_animate.py \
#     -i "datasets/benchmark_dataset/person_1" \
#     --name "person_1" \
#     -o "photomakeradapter_person_1" \
#     --ip_adapter \
#     --use_clipl_embed

# python test_animate.py \
#     -i "datasets/benchmark_dataset/person_3" \
#     --name "person_3" \
#     -o "photomaker_person_3"

python test_animate.py \
    -i "datasets/benchmark_dataset/person_3" \
    --name "person_3" \
    -o "photomakeradapter_clipG_person_3" \
    --ip_adapter

python test_animate.py \
    -i "datasets/benchmark_dataset/person_3" \
    --name "person_3" \
    -o "photomakeradapter_person_3" \
    --ip_adapter \
    --use_clipl_embed