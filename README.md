# Study Log
Checklist to help me keep track of milestones in a study plan designed to help familiarize myself with the fundementals of ML (particularly deep learning) theory. Will probably be expanded to cover other domains of knowledge in software engineering.

Inspiration for this log: https://github.com/amitness/learning

# AI Study Plan Resources Checklist

## Foundations
**Purpose**: Build a strong foundation in Linear Algebra, Calculus, Statistics, Neural Networks, and AI frameworks.

### Foundational Mathematics Resources

|Resource|Progress|Notes|
|---|---|---|
|[Imperial College London, MOOC: Mathematics for Machine Learning - Specialization, Linear Algebra](https://www.coursera.org/learn/linear-algebra-machine-learning/)|✅|
|[Imperial College London, MOOC: Mathematics for Machine Learning - Specialization, Multivariant Calculus](https://www.coursera.org/specializations/mathematics-machine-learning)|✅|
|[Imperial College London, MOOC: Mathematics for Machine Learning - Specialization, Linear Algebra](https://www.coursera.org/specializations/mathematics-machine-learning)|⬜|
|MIT OpenCourseWare: [Linear Algebra Full Course](https://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/) |⬜ |for deepening knowledge/review|

### Programming Resources and General Tools/Setup
|Resource|Progress|
|---|---|
|Google Colab for experimentation: [Colab](https://colab.research.google.com/)|⬜|
|[PyTorch 60-Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)|⬜|


### Machine Learning/Neural Networks Foundational Resources
|Resources|Progress|Notes|
|---|---|---|
|[LinkedIn Learning: Building Computer Vision Applications with Python](https://www.linkedin.com/learning/building-computer-vision-applications-with-python/computer-vision-under-the-hood)|✅|turned out to be more of a graphics processing course, but at least provides a good intro to convolution filters (just does not actually provide the whole picture with regard to convolutional neural network architecture)|
|[Andrew Ng: Neural Networks and Deep Learning](https://www.coursera.org/learn/neural-networks-deep-learning?specialization=deep-learning)|✅||
|[Andrew Ng: Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization](https://www.coursera.org/learn/deep-neural-network?specialization=deep-learning)|✅||
|[Andrew Ng: Structuring Machine Learning Projects](https://www.coursera.org/learn/machine-learning-projects?specialization=deep-learning)|✅| |
|[Andrew Ng: Convolutional Neural Networks](https://www.coursera.org/learn/convolutional-neural-networks?specialization=deep-learning)|✅| |
|[Andrew Ng: Sequence Models](https://www.coursera.org/learn/nlp-sequence-models?specialization=deep-learning)|✅|Found this sequence of deep learning courses overall extremely illuminating. Definititely leaves you with enough of an understanding of LSTMs/GRUs, ConvNets, and ResNets to start tinkering confidently. Final module on tranformers felt pretty abstract/rushed compared to every other module.|
|[3Blue1Brown: Transformers (how LLMs work) explained visually](https://www.youtube.com/watch?v=wjZofJX0v4M)|✅| Youtube video. Useful followup for final module of Andrew Ng's "Sequence Models" course. |
|[Andrej Karpathy: Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY)|⬜|youtube video|
|[Neural machine translation with a Transformer and Keras](https://www.tensorflow.org/text/tutorials/transformer)|⬜| article from tensorflow docs|
|[Fine-tune a pretrained (transformer) model](https://huggingface.co/docs/transformers/training)|⬜|huggingface docs|
|[Textbook: *Deep Learning* by Goodfellow et al. (online version)](https://www.deeplearningbook.org/) |⬜|textbook|

### Topics to follow up on/review from ML courses:
- Understanding the properties/behavior of commonly used loss functions
- Transformer architecture
- Math review exercise: walk through backprop for a Sequential model with GRUs  or LSTM cells

---

## ML Papers

|Resource|Progress|Notes|
|---|---|----|
|[Attention is all you need](https://arxiv.org/abs/1706.03762)|⬜| (Introduced transformer architecture)||
|[Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903?utm_campaign=The%20Batch&utm_source=hs_email&utm_medium=email&_hsenc=p2ANqtz-9kgISsDFvuO8MZlBnFosRC4C4FiNqno6ahMESpHrnRkOKvDeon1AkJ43ZnkA-hwbA6vq6q)|⬜||
|[LIPNET: END-TO-END SENTENCE-LEVEL LIPREADING](https://arxiv.org/pdf/1611.01599)|✅| Authors: Yannis M. Assael, Brendan Shillingford, Shimon Whiteson, & Nando de Freitas. Sentence-level data set improves accuracy of the model (~95%), as context is extremely helpful in disambiguating visemes (visual phonemes) that look confusingly similar. Techniques: CTC Loss, STCNN, bidirectional GRU, beam search.|
|[Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks](https://www.cs.toronto.edu/~graves/icml_2006.pdf)|⬜| Graves, Fernandez, Gomez, Schmidhuber (2006). Followup on CTC Loss for LipNet paper.|
|[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)|⬜| Deep Learning 4 course, supplemental recommended reading. Authors: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun|
|[You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)|⬜|Deep Learning 4 course, supplemental recommended reading. Authors: Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi|
|[YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242)|⬜|Deep Learning 4 course, supplemental recommended reading. Authors: Joseph Redmon, Ali Farhadi|
|[MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)|⬜|Deep Learning 4 course, supplemental recommended reading. Authors: Sandler, Howard, Zhu, Zhmoginov & Chen (2018)|
|[EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)|⬜|Deep Learning 4 course, supplemental recommended reading. (Tan & Le, 2019)|
|[Fully Convolutional Architectures for Multi-Class Segmentation in Chest Radiographs](https://arxiv.org/abs/1701.08816)|⬜|Deep Learning 4 course, supplemental recommended reading. (Novikov, Lenis, Major, Hladůvka, Wimmer & Bühler, 2017) |
|[Automatic Brain Tumor Detection and Segmentation Using U-Net Based Fully Convolutional Networks](https://arxiv.org/abs/1705.03820)|⬜|Deep Learning 4 course, supplemental recommended reading. (Dong, Yang, Liu, Mo & Guo, 2017) |
|[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)|⬜|Deep Learning 4 course, supplemental recommended reading. (Ronneberger, Fischer & Brox, 2015) |
|[FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/pdf/1503.03832)|⬜|Deep Learning 4 course, supplemental recommended reading. (Schroff, Kalenichenko & Philbin, 2015)|
|[DeepFace: Closing the Gap to Human-Level Performance in Face Verification](https://scontent-lga3-2.xx.fbcdn.net/v/t39.8562-6/240890413_887772915161178_4705912772854439762_n.pdf?_nc_cat=109&ccb=1-7&_nc_sid=e280be&_nc_ohc=NdyjNPFINoMQ7kNvgHCWM0U&_nc_oc=AdgRMRncPbzdvh37g-NoHuwz6yO_jwAcyL6st3OIzMKsdNxPwOTkK7J64RFkSlRbnNigrA4en__zp7kTVjkzlvnK&_nc_zt=14&_nc_ht=scontent-lga3-2.xx&_nc_gid=ANzXYunySZDsZuTUA76pLsD&oh=00_AYAM7rvBxNpSahCTFZsFf6ot_C_jnLFDvg9TJFbD3AZ5Ig&oe=67B6D77F)|⬜|Deep Learning 4 course, supplemental recommended reading. (Taigman, Yang, Ranzato & Wolf) |
|[A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)|⬜|Deep Learning 4 course, supplemental recommended reading. (Gatys, Ecker & Bethge, 2015)|
|[Very Deep Convolutional Networks For Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf)|⬜|Deep Learning 4 course, supplemental recommended reading. (Simonyan & Zisserman, 2015)|
|[Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings](https://proceedings.neurips.cc/paper_files/paper/2016/file/a486cd07e4ac3d270571622f4f316ec5-Paper.pdf)|⬜|Deep Learning 4 course, supplemental recommended reading. (Bolukbasi et al., 2016)|
|[Learning Object Permanence from Videos via Latent Imaginations](https://arxiv.org/pdf/2310.10372)|⬜|(Traub, Becker, Otte, Butz, 2024) Curious about learning processes for spatial data, as this is something that human brains are so good at, it can be used as a mnemonic technique (e.g., method of loci). This seems like it might be tangentially related.|
|STAM-SNN: Spatio-Temporal Associative Memory in Brain-Inspired Spiking Neural Networks: Concepts and Perspectives|⬜|have local pdf of preprint|

---


### Open Source and Community
|Resource|Progress|
|---|---|
|Kaggle Datasets: [Kaggle Datasets](https://www.kaggle.com/datasets)|⬜|
---


# System Design Study Plan

## General Resources
|Resource|Progress|Notes|
|---|---|---|
|[Designing Data Intensive Applications](https://www.oreilly.com/library/view/designing-data-intensive-applications/9781491903063/) |⬜||
|[https://docs.djangoproject.com/en/5.1/intro/](https://docs.djangoproject.com/en/5.1/intro/)|✅||