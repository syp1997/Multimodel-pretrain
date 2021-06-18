## Multimodel

### M6: A Chinese Multimodal Pretrainer

Link: <https://arxiv.org/pdf/2103.00823.pdf>

Abstract: M6-Corpus, 1.9TB images and 292GB texts; Downstream applications: product description generation, visual question answering, community question answering, Chinese poem generation and text-to-image generation; large-scale distributed training optimizations.

Contribution: Construct a large dataset, containing both plain text and image-text pairs.

Pretrain method: Text-to-text, Image-to-text, Multimodality-to-text

Dataset: M6

Downstream Task: Text-to-Image Generation(50 million product titles and images), Visual Question Answering(FMIQA dataset), Image Captioning(E-Commerce IC dataset), Question Answering(various Chinese forums), Poem Generation, Image-Text Matching(E-Commerce ITM dataset)

### SemVLP: Vision-Language Pre-training by Aligning Semantics at Multiple Levels

Link: <https://arxiv.org/pdf/2103.07829.pdf>

Abstract: Jointly **aligns both the low-level and highlevel** semantics between image and text representations, pre-trained iteratively with two prevalent fashions, single-stream pre-training and two-stream pre-training.

Pretrain method: Masked LM Prediction, Masked Object Prediction, Image-Text Matching, Image Question Answering

Dataset: MS COCO, Visual Genome (image caption);  VQA v2.0, GQA balanced version and VG-QA(image question answering);  Conceptual Captions and SBU Captions(4M image-text pairs);  

Downstream Task: Image Question Answering(VQA v2.0, GQA 2019), Image-Text Retrieval(Flickr30K dataset), visual reasoning task(NLVR2)

### WenLan: Bridging Vision and Language by Large-Scale Multi-Modal Pre-Training

Link: <https://arxiv.org/pdf/2103.06561.pdf>

Abstract: A **two-tower** pre-training model called BriVL within the **crossmodal contrastive learning** framework, adapting the latest method MoCo into the cross-modal scenario. Construct Chinese multi-source imagetext dataset called RUC-CAS-WenLan. 

Pretrain method: image-text retrieval task, to learn two encoders that can embed image and text samples into the same space.

Dataset: RUC-CAS-WenLan(30 million image-text pairs) and AIC-ICC.

Downstream task: MatchSoul and Soul-Music

### VL-BERT: Pre-training of Generic Visual-Linguistic Representations

Link: <https://arxiv.org/pdf/1908.08530.pdf>

GitHub: <https://github.com/jackroos/VL-BERT>

Abstract: Pretrain VL-BERT at both large visual-linguistic corpus and text-only datasets, each element is either of a word from the input sentence, or a region-of-interest (RoI) from the input image.

Pretrain method: Masked Language Modeling with Visual Clues, Masked RoI Classification with Linguistic Clues

Dataset: Conceptual Captions, BooksCorpus and English Wikipedia

Downstream task: Visual Commonsense Reasoning(VCR dataset), Visual Question Answering(VQA v2.0 dataset), Referring Expression Comprehension(RefCOCO+ dataset)

### VideoBERT: A Joint Model for Video and Language Representation Learning

Link: <https://arxiv.org/pdf/1904.01766.pdf>

Github: <https://github.com/ammesatyajit/VideoBERT>

Abstract: Build upon the BERT model to learn bidirectional joint distributions over sequences of visual and linguistic tokens, derived from vector quantization of video data and off-the-shelf speech recognition outputs.

Contribution: A simple way to learn high level video representations that capture semantically meaningful and temporally long-range structure.

Pretrain method: video and text masked token prediction, linguistic-visual alignment task

Dataset: Cooking312K, YouCook II

Downstream task: Zero shot action classification(YouCook II), Transfer learning for captioning(YouCook II), text-to-video generation and future forecasting

<div align="center">    
<img src="https://github.com/syp1997/Multimodel-pretrain/blob/main/imgs/VideoBERT.png" alt="VideoBERT" width = "100%" height="100%"/>
</div>

### Learning Video Representations using Contrastive Bidirectional Transformer

Link: <https://arxiv.org/pdf/1906.05743.pdf>

Abstract: Self-supervised learning approach for video features, two-stream, Cross-modal learning, use noise contrastive estimation (NCE) loss. 

Pretrain method: L<sub>cbt</sub> = w<sub>bert</sub> L<sub>bert</sub> + w<sub>visual</sub>L<sub>visual</sub> + w<sub>cross</sub>L<sub>cross</sub>, fix w<sub>bert</sub> = 0. Feature extraction or fine-tune.

Dataset: Kinetics, HowTo100M

Downstream task: **Action recognition(UCF101, HMDB51)**, Action anticipation(Breakfast dataset, 50Salads dataset, ActivityNet 200 dataset), **Video captioning(YouCook2)**, **Action segmentation(COIN dataset)**.

<div align="center">    
<img src="https://github.com/syp1997/Multimodel-pretrain/blob/main/imgs/CBT.png" alt="CBT" width = "100%" height="100%"/>
</div>

### UniVL: A Unified Video and Language Pre-Training Model for Multimodal Understanding and Generation

Link: <https://arxiv.org/pdf/2002.06353.pdf>

GitHub: <https://github.com/microsoft/UniVL>

Abstract: video-linguistic pre-training, including two single-modal encoders, a cross encoder, and a decoder. Use S3D to extract video feature.

Pretrain method: Five objectives: video-text joint, conditioned masked language model, conditioned masked frame model, video-text alignment, and language reconstruction. Two pre-training strategies: only preserve the text BERT and video Transformer to learn the weights using the Video-Text Joint loss; mask the whole text tokens with a 15% possibility.

Dataset: Howto100M

Downstream task: **text-based video retrieval(Youcook2, MSR-VTT)**, **multimodal video captioning(Youcook2)**, **action segmentation(COIN dataset)**, action step localization(CrossTask), and **multimodal sentiment analysis(CMU-MOSI)**.

<div align="center">    
<img src="https://github.com/syp1997/Multimodel-pretrain/blob/main/imgs/UniVL.png" alt="UniVL" width = "100%" height="100%"/>
</div>

### ActBERT: Learning Global-Local Video-Text Representations

Link: <https://arxiv.org/pdf/2011.07231.pdf>

Abstract: First, incorporates global actions, local regional objects and text descriptions in a joint framework. Second, introduce a TaNgled Transformer block(TNT) to encode features from these three sources. Single stream, four different embeddings, position embedding, segment embedding, token embedding, visual feature embedding. 3D CNN to extract action features, faster R-CNN to extract regional object features.

Pretrain method: 

Downstream task: text-video clip retrieval, video captioning, video question answering, action segmentation, and action step localization.
