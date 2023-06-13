<!--
Copyright (C) 2023 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: MIT
-->


# Dependent pre-trained models
Our code used the following publicly available pre-trained models for visual processing, language modeling, and vision-and-language reasoning. We also provide below the licenses associated with these models, and the download links that we used. Note that we do not use or modify the code behind these pre-trained models, and one may use other implementations of these models, if needed.

| Name               | License        | Link |
|:-------------------|:---------------|------|
| BERT/HuggingFace   | Apache-2.0     | https://huggingface.co/docs/transformers/model_doc/bert |
| GPT2/HuggingFace   | MIT            | https://huggingface.co/gpt2 |
| MAE/HuggingFace    | Apache-2.0     | https://huggingface.co/facebook/vit-mae-large |
| CrossTransformer   | MIT            | https://github.com/lucidrains/vit-pytorch |
| CLIP/OpenAI        | MIT            | https://github.com/openai/CLIP |
| FLAVA/HuggingFace  | BSD-3-Clause   | https://huggingface.co/facebook/flava-full |

We also used the following models that are part of the torchvision toolbox of PyTorch (released under `BSD-3-Clause`)
| Name               | License        | Link |
|:-------------------|:---------------|------|
| AlexNet/VGG        | BSD-3-Clause   | https://pytorch.org/vision/stable/models.html |
| ResNet-50          | BSD-3-Clause   | https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html |
| ResNet-18          | BSD-3-Clause   | https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html |
| ViT-16             | BSD-3-Clause   | https://pytorch.org/vision/main/models/vision_transformer.html |
| Swin_b, Swin_t     | BSD-3-Clause   | https://pytorch.org/vision/main/models/swin_transformer.html |

We used GloVe language embeddings from the torchtext toolbox of PyTorch.
| Name               | License        | Link |
|:-------------------|:---------------|------|
| GloVe from TorchText | BSD-3-Clause | https://pypi.org/project/torchtext/ |
