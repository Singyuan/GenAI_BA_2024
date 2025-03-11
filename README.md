# GenAI_BA_2025
## Objectives of "TA class"
#### Mens et Manus -- MIT motto
**Goal 1: Understand the Problem**
Use your "mind" to analyze and understand the essence of the problem. This reflects the concept of "Mens."

**Goal 2: Search the Solution**
Use your "mind" to search for knowledge and analyze possible solutions, finding the best approach.

**Goal 3 (option): Implement the Solution**
Put your knowledge into action by solving the problem through practical execution. This embodies the idea of "Manus."
![deepseek](https://hackmd.io/_uploads/BJaN-_uFye.png)

### Roadmap
Retrieval Augmented Generation (RAG) is a language model (LM) that uses an external datastore at test time.
![arch](https://hackmd.io/_uploads/rk3wpvzYJx.png)

**The master of RAG roadmap**
![roadmap](https://hackmd.io/_uploads/rJtFnJGFJg.png)

## Calendar

|Week|Date|Contents|TA class|
|-|-|-|-|
|1|2/19|Open & Prof. Chung (NTUT)|x|
|2|2/26|Dr. Lee (NVIDIA)|x|
|3|3/5|Dr. Lee (NVIDIA)|x|
|4|3/12|Dr. Wu (SYSTEX)||
|5|3/19|SYSTEX||
|6|3/26|SYSTEX||
|7|4/2|SYSTEX|?|
|8|4/9|SYSTEX||
|9|4/16|Dr. Wu (SYSTEX)||
|10|4/23|Proposal|x|
|11|4/30|Prof. Yang (NTU)||
|12|5/7|Prof. Yang (NTU)||
|13|5/14|Prof. Yang (NTU)||
|14|5/21|Catch up|?|
|15|5/28|Presentation|x|



## Contents
|#|Date|Topic|Contents|Hands-on|Note|
|-|-|-|-|-|-|
|<a href="#Preliminary">0</a>|3/12|Preliminary|<ul><li>Intro 2 Pytorch</li><li>Intro 2 DL</li><li>DL model</li></ul>|[DL model](https://colab.research.google.com/drive/1Ai5rEIYcpqYiLgdNIkdmsc47QWLvcpbt#scrollTo=V57zhcTp1Xxb)||
|<a href="#small-Language-Models">1</a>|3/19|sLM|<ul><li>Transformer</li><li>BERT</li><li>Tokenization</li></ul>|[Transformer](https://colab.research.google.com/drive/1YUqcXCoP9RAaykSIz3IC2Bz3j6OTSawF?usp=sharing)||
|<a href="#Large-Language-Models-LLMs">2</a>|3/26|LLMs|<ul><li>Training steps of LLMs</li><li>Evolution of LLMs</li><li>Quantization</li></ul>|<ul><li>[Call by API](https://colab.research.google.com/drive/1wwZp7Y3jZQMawKd523CJCd-u0VFIMGs0?hl=zh-tw#scrollTo=z4I00KtDfDs3)</li><li>[Quantization]()</li><ul>||
|<a href="#Improve-PLMs-Fine-tuning">3</a>||Improve PLMs|<ul><li>Introduction to different FT ways</li><li>Comparison to different FT ways</li></ul>|[Fine-tuning]()||
  |<a href="#Retrieval-Augmented-Generation-RAG">4</a>|4/16|RAG|<ul><li>How to retrieve</li><li>When to retrieve</li><li>What to retrieve</li><li>How to use retrieve</li></ul>|[RAG](https://colab.research.google.com/drive/1s1nlPUIG0fGK4VSHRH8pR3JiEpsZRfFO?usp=sharing)||

## Contact us
Sing-Yuan Yeh: d10948003@ntu.edu.tw  
Jhe-Jia Wu: d11948002@ntu.edu.tw

## Preliminary
- **Optimization**: The process of adjusting model parameters to minimize or maximize a certain objective function.
    - Adam
    - RMSProp
- **Forward and backward propagation**: The mechanism for computing the output of a neural network and updating its weights.
- **Loss function**: A function that measures the difference between predicted outputs and true labels.
    - Mean Squared Error (MSE)
    - Cross-Entropy Loss
- **Activation function**: Functions applied to the output of each neuron to introduce non-linearity into the network.
    - ReLU (Rectified Linear Unit):
    - sigmoid
    - LeakyReLu
- **Regularization**: Techniques like L1, L2 regularization, or dropout used to prevent overfitting by adding penalties or modifying the network during training.
- **Neural Network Architecture**: The structure or layout of a neural network, which defines how layers of neurons are connected.
    - Fully Connected (Dense) Layer
    - Convolutional Layer
    - Batch Normalization Layer
    - Residual
- **Epoch**: One full pass through the entire training dataset during the training process.
- **Batch Size**: The number of training samples used in one forward and backward pass through the network before updating the weights.
- **Initial Value (Weight Initialization)**: The process of setting the initial values for the weights of the neural network before training begins.

## small-Language-Models
### Tokenization 
- **WordPiece**: Tokenizes words into subword units based on likelihood. Used by: BERT and related models.
- **Byte Pair Encoding (BPE)**: Tokenizes text into subwords by iteratively merging the most frequent pair of bytes. Used by: GPT-2, Transformer models. Please refer to [BPE](https://platform.openai.com/tokenizer).
- **Unigram Language Model**: Tokenizes words into subwords based on a probabilistic model that selects the most likely subword units. Used by: T5.

### Self-Attention
The third character is Bertolt, whose name contains "BERT". Hence, the word "BERT" seems to be associated with something large.

![DALLE2025-02-0612.13.11-Asequenceoffourfictionalcreaturesarrangedfromlefttorightincreasinginsizeexponentially.Thefirstcreatureontheleftisamechanicalr-ezgif.com-optipng-min](https://hackmd.io/_uploads/rk8MGlGKke.png)
> Created by ChatGPT

xfomer family
- [Transformer](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf): Google 2017.
- [GPT (Generative Pre-trained Transformer)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) OpenAI 2018/6.
- [BERT (Bidirectional Encoder Representations from Transformers)](https://arxiv.org/abs/1810.04805): OpenAI 2018/10.
- [T5 (Text to Text Transfer Transformer)](https://arxiv.org/abs/1910.10683) Google 2019
- [Conformer (Convolution-augmented Transformer)](https://arxiv.org/abs/2005.08100)
- [Longformer (Long-Document Transformer)](https://arxiv.org/abs/2004.05150)
- [Big Bird (Transformers for Longer Sequences)](https://arxiv.org/abs/2007.14062)
- [Reformer](https://arxiv.org/abs/2001.04451)