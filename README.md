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
![arch2](https://hackmd.io/_uploads/HkHJ1D_TJl.png)

**The master of RAG roadmap**
![rag_class2](https://hackmd.io/_uploads/r1GmPHu6yx.png)

## Calendar

|Week|Date|Contents|TA class|
|-|-|-|-|
|1|2/19|Open & Prof. Chung (NTUT)|x|
|2|2/26|Dr. Lee (NVIDIA)|x|
|3|3/5|Dr. Lee (NVIDIA)|x|
|4|3/12|Dr. Wu (SYSTEX)||
|5|3/19|SYSTEX||
|6|3/26|SYSTEX||
|7|4/2|SYSTEX||
|8|4/9|SYSTEX|*|
|9|4/16|Dr. Wu (SYSTEX)||
|10|4/23|Proposal|x|
|11|4/30|Prof. Yang (NTU)||
|12|5/7|Prof. Yang (NTU)||
|13|5/14|Prof. Yang (NTU)||
|14|5/21|Catch up|?|
|15|5/28|Presentation|x|

\* The TA will be in the office. If you have any questions, feel free to come to the office and ask us.

## Contents
|#|Date|Topic|Contents|Hands-on|Note|
|-|-|-|-|-|-|
|<a href="#Preliminary">0</a>|3/12|Preliminary|<ul><li>Intro 2 Pytorch</li><li>Intro 2 DL</li><li>DL model</li></ul>|<ul><li>[Training process](https://colab.research.google.com/drive/1QX-WMvQIsZAaRPWCaaFZquKj0OrUaclY?usp=sharing)</li><li>[DL model](https://colab.research.google.com/drive/1Ai5rEIYcpqYiLgdNIkdmsc47QWLvcpbt#scrollTo=V57zhcTp1Xxb)</li></ul>|
|<a href="#small-Language-Models">1</a>|3/19|sLMs|<ul><li>Transformer</li><li>BERT</li><li>Tokenization</li></ul>|[Transformer](https://colab.research.google.com/drive/1YUqcXCoP9RAaykSIz3IC2Bz3j6OTSawF?usp=sharing)||
|<a href="#Large-Language-Models-LLMs">2</a>|3/26|LLMs|<ul><li>Training steps of LLMs</li><li>Evolution of LLMs</li><li>Quantization</li></ul>|<ul><li>[Call by API](https://colab.research.google.com/drive/1wwZp7Y3jZQMawKd523CJCd-u0VFIMGs0?hl=zh-tw#scrollTo=z4I00KtDfDs3)</li><li>[Quantization](https://colab.research.google.com/drive/1bYs4uz6vE0Amx1-4d3wMsFfd_-Ci82LQ?usp=sharing)</li><ul>||
|<a href="#Improve-PLMs">3</a>|4/2|Improve PLMs|<ul><li>Introduction to different FT ways</li><li>Comparison to different FT ways</li></ul>|[Fine-tuning](https://colab.research.google.com/drive/1vQbcYXRVrZz_NGuaxr3bxQ9wFDaBfVBC?usp=sharing)||
  |<a href="#Retrieval-Augmented-Generation-RAG">4</a>|4/16|RAG|<ul><li>How to retrieve</li><li>When to retrieve</li><li>What to retrieve</li><li>How to use retrieve</li></ul>|[RAG](https://colab.research.google.com/drive/12kBCiWkCnfDcOHHrJX0BMx7hhthknvQw?usp=sharing)||

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
![training](https://hackmd.io/_uploads/BkWLuMNhye.png)

### Supplementary
[Neural Tangent Kernel](https://colab.research.google.com/drive/1JzUrh1k1xDC03JZX5HlExsIDSkkpzXZc?usp=sharing)

## small-Language-Models

### Tokenization 
- **WordPiece**: Tokenizes words into subword units based on likelihood. Used by: BERT and related models.
- **Byte Pair Encoding (BPE)**: Tokenizes text into subwords by iteratively merging the most frequent pair of bytes. Used by: GPT-2, Transformer models. Please refer to [BPE](https://platform.openai.com/tokenizer).
- **Unigram Language Model**: Tokenizes words into subwords based on a probabilistic model that selects the most likely subword units. Used by: T5.

### Self-Attention
The third character is Bertolt, whose name contains "BERT". Hence, the word "BERT" seems to be associated with something large.

![DALLE2025-02-0612.13.11-Asequenceoffourfictionalcreaturesarrangedfromlefttorightincreasinginsizeexponentially.Thefirstcreatureontheleftisamechanicalr-ezgif.com-optipng-min](https://hackmd.io/_uploads/rk8MGlGKke.png)
> Created by ChatGPT

xxxfomer family
- [Transformer](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf): Google 2017.
- [GPT (Generative Pre-trained Transformer)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) OpenAI 2018/6.
- [BERT (Bidirectional Encoder Representations from Transformers)](https://arxiv.org/abs/1810.04805): OpenAI 2018/10.
- [T5 (Text to Text Transfer Transformer)](https://arxiv.org/abs/1910.10683) Google 2019
- [Conformer (Convolution-augmented Transformer)](https://arxiv.org/abs/2005.08100)
- [Longformer (Long-Document Transformer)](https://arxiv.org/abs/2004.05150)
- [Big Bird (Transformers for Longer Sequences)](https://arxiv.org/abs/2007.14062)
- [Reformer](https://arxiv.org/abs/2001.04451)


## Large-Language-Models-LLMs
![1-cbfa24f7 (1)](https://hackmd.io/_uploads/SkT_G51K1g.png)
> Created by ChatGPT

### Training steps of LLM
  ![3step](https://hackmd.io/_uploads/B1-uSgMtyx.png)

  - [Step 1](https://drive.google.com/file/d/1myvHjoeFOpIl1uGU9H1t4OpDErkhF0zO/view): [Pre-trained method](https://arxiv.org/abs/1810.04805) e.g. Masked languege modeling, Next sentence prediction. 
  - [Step 2](https://drive.google.com/file/d/1SOXBQhsC_L6aHXcLx2rltaDdcO6N2FmJ/view): Fine-tuning (next section)
  - [Step 3](https://speech.ee.ntu.edu.tw/~hylee/genai/2024-spring-course-data/0412/0412_LLMtraining_part3.pdf): real-world or RL practice

### Three types of Pre-Training model (PLM)
 Please refer to the [Survey paper](https://arxiv.org/abs/2402.06196) for more details.

  |type|Example |
  |-|-|
  |Encoder|[BERT](https://arxiv.org/abs/1810.04805)|
  |Decoder|<ul><li>[GPT-3 (Generative Pre-trained Transformer)](https://papers.nips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html)</li><li>[LLaMA1 (Large Language Model Meta AI)](https://arxiv.org/abs/2302.13971)</li><li>[Mistral](https://arxiv.org/abs/2310.06825)</li></ul> |
  |Encoder-decoder|<ul><li>[Transformer](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)</li> <li>[T5 (Text to Text Transfer Transformer)](https://arxiv.org/abs/1910.10683)</li></ul>|


### Evolution

|Stage|Time|Model |Publisher|Constribution|# of parms|
  |-|-|-|-|-|-|
  |1|2017/6|[Transformer](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)|Google|Originator||
  |1|2018/6|GPT|OpenAI||117M|
  |1|2018/10|[BERT](https://arxiv.org/abs/1810.04805)|Google|Pre-trained methods|340M|
  |1|2019/2|GPT-2|OpenAI||1.5B|
  |2|2019/10|[T5](https://arxiv.org/abs/1910.10683)|Google||11B|
  |2|2020/5|[GPT-3](https://papers.nips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html)|OpenAI|Prompt learning|175B|
  |3|2022/4|[PaLM](https://arxiv.org/abs/2204.02311)|Google|[Chain-of-Thought](https://arxiv.org/abs/2201.11903)|540B|
  |3|2022/12|ChatGPT|OpenAI|Based on gpt-3|
  |3|2023/2|[LLaMA](https://arxiv.org/abs/2302.13971)|Meta||7B to 65B|
  |4|2023/7|LLaMA2|Meta|Open source|7B to 70B|
  |4|2023/12|Gemini|Google||
  |4|2024/2|Sora|OpenAI|Multimodal models|
  |4|2025/1|DeepSeek-R1|DeepSeek||1.5B to 671B|
  |4|2025/2|Grok3|X||

### Quantization
- `FP32`
- `BF16`
- `int8()`
- `uint8()`
- `NF4`

Please refer to [mlabonne blog](https://mlabonne.github.io/blog/posts/Introduction_to_Weight_Quantization.html), [HF](https://huggingface.co/blog/hf-bitsandbytes-integration) or [bnb](https://huggingface.co/docs/bitsandbytes/index) for more details.


## Improve PLMs
  ![FT](https://hackmd.io/_uploads/SkHxYeftyx.png)

Because meta has released a pretrained large language model LLaMA, everyone only needs to use fine-tuning to create our own model. That's why Prof. Lee said, "舊時王謝堂前llama 飛入尋常百姓家." 
  ![截圖 2025-02-05 凌晨2.41.41](https://hackmd.io/_uploads/r1rtk1lFyg.png)
  > [Image source](https://arxiv.org/abs/2303.18223)

  Here are some examples of current Traditional Chinese models in Taiwan, along with the models they were fine-tuned on.
  |#|Step 1|Step 2|
|-|-|-|
|0|LLaMA|[Taide (NCHC)](https://taide.tw/index)|
|1|Mistral|[Breeze (MTK Research)](https://www.mediatek.tw/blog/mediatek-research-breeze-7b)|


#### Supervised Fine-Tuning with LoRA
  Supervised Fine-Tuning (SFT) adapts pre-trained models to specific tasks using labeled data, improving performance by adjusting the entire model. LoRA is an efficient fine-tuning approach within SFT that reduces computational and storage costs by introducing low-rank matrices. [Paper](https://arxiv.org/abs/2106.09685)

## Retrieval-Augmented-Generation-RAG
![rag_class3](https://hackmd.io/_uploads/HycaAIdp1x.png)
1. Splitter: Splits text into smaller chunks for easier processing. [LangChain Document](https://python.langchain.com/docs/concepts/text_splitters/)
2. Embedding: Converts text into numerical vectors that capture meaning. [LangChain Document](https://python.langchain.com/docs/concepts/embedding_models/)
3. Storage: Stores and organizes vector data for efficient retrieval. [LangChain Document](https://python.langchain.com/docs/concepts/vectorstores/)
4. Retriver: Retrieves relevant data or documents based on queries. [LangChain Document](https://python.langchain.com/docs/concepts/retrievers/)

- Flow chart image from `LangChain` [Document](https://js.langchain.com/v0.1/docs/modules/chains/document/stuff/).
![](https://js.langchain.com/v0.1/assets/images/stuff-818da4c66ee17911bc8861c089316579.jpg)

- **Storage**
  - vector databases
  - [graph database](https://arxiv.org/abs/2408.08921)
  
- **Retriever (How to retrieve?)**
  - Sparse retrieval: TF-IDF, BM25
  - Dense retrieval: [DPR](https://arxiv.org/abs/2004.04906), Contriever

- **What to retrieve? How to use retrieve? When to retrieve?**
  
| Model | What do retrieve? | How to use retrieval? | When to retrieve? |
|-----------------------------|---------------------|-------------------|-----------------|
| [RAG](https://arxiv.org/abs/2005.11401) (Lewis et al 2020) | Text chunks | Input layer | Once |
| Retrieve-in-context LM (Shi et al 2023, Ram et al 2023) | Text chunks | Input layer | Every n tokens |
| RETRO (Borgeaud et al. 2021) | Text chunks | Intermediate layers | Every n tokens |
| kNN-LM (Khandelwal et al. 2020) | Tokens | Output layer | Every token |
| FLARE (Jiang et al. 2023) | Text chunks | Input layer | Every n tokens *(adaptive)* |


### Summary: Fine-tuning v.s RAG:

  ![截圖 2025-02-05 下午1.46.06](https://hackmd.io/_uploads/SkTshdeK1g.png)
  > [Image source](https://arxiv.org/abs/2312.10997)


## References
  - [CMU](https://www.phontron.com/class/anlp2024/assets/slides/anlp-10-rag.pdf)
  - [ACL](https://acl2023-retrieval-lm.github.io/slides/3-architecture.pdf)