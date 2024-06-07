# python test_animate.py \
#     -i "examples/scarletthead_woman" \
#     -p "woman.txt" \
#     --name "scarletthead" \
#     -o "photomaker_scarletthead"

python test_animate.py \
    -i "examples/scarletthead_woman" \
    -p "woman.txt" \
    --name "scarletthead" \
    -o "photomakeradapter_clipG_scarletthead" \
    --ip_adapter

python test_animate.py \
    -i "examples/scarletthead_woman" \
    -p "woman.txt" \
    --name "scarletthead" \
    -o "photomakeradapter_scarletthead" \
    --ip_adapter \
    --use_clipl_embed


# python test_animate.py \
#     -i "examples/yangmi_woman" \
#     -p "woman.txt" \
#     --name "yangmi" \
#     -o "photomaker_yangmi"

# python test_animate.py \
#     -i "examples/yangmi_woman" \
#     -p "woman.txt" \
#     --name "yangmi" \
#     -o "photomakeradapter_clipG_scarletthead" \
#     --ip_adapter

# python test_animate.py \
#     -i "examples/yangmi_woman" \
#     -p "woman.txt" \
#     --name "yangmi" \
#     -o "photomakeradapter_yangmi" \
#     --ip_adapter \
#     --use_clipl_embed

# python test_animate.py \
#     -i "examples/newton_man" \
#     --name "newton" \
#     -o "photomaker_newton"

# python test_animate.py \
#     -i "examples/newton_man" \
#     --name "newton" \
#     -o "photomakeradapter_clipG_newton" \
#     --ip_adapter

# python test_animate.py \
#     -i "examples/newton_man" \
#     --name "newton" \
#     -o "photomakeradapter_newton" \
#     --ip_adapter \
#     --use_clipl_embed