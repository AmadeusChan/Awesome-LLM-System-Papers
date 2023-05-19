# Awesome-LLM-System-Papers

This is a list of (non-comprehensive) LLM system papers maintained by [ALCHEM Lab](https://alchem.cs.purdue.edu/index.html). Welcome to create a pull requst or an issue if we have missed any interesting papers!

## Algorithm-System Co-Design

- Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity (JMLR'21) [link to paper](https://www.jmlr.org/papers/volume23/21-0998/21-0998.pdf)
- Scalable and Efficient MoE Training for Multitask Multilingual Models (arXiv'21) [link to paper](https://arxiv.org/pdf/2109.10465.pdf)
- DeepSpeed-MOE: Advancing Mixture of Experts Inference and Training to Power Next-Generation AI Scale (ICML'22) [link to paper](https://proceedings.mlr.press/v162/rajbhandari22a/rajbhandari22a.pdf)

## LLM Inference (Serving) Systems

### Single-GPU Systems

- TurboTransformers: An Efficient GPU Serving System For Transformer Models (PPoPP'21) [link to paper](https://dl.acm.org/doi/pdf/10.1145/3437801.3441578)
- PetS: A Unified Framework for Parameter-Efficient Transformers Serving (ATC'22) [link to paper](https://www.usenix.org/system/files/atc22-zhou-zhe.pdf)

### Distributed Systems

- Orca: A Distributed Serving System for Transformer-Based Generative Models (OSDI'22) [link to paper](https://www.usenix.org/conference/osdi22/presentation/yu)
- DeepSpeed-inference: enabling efficient inference of transformer models at unprecedented scale (SC'22) [link to paper](https://dl.acm.org/doi/abs/10.5555/3571885.3571946)
- EnergeonAI: An Inference System for 10-100 Billion Parameter Transformer Models (arXiv'22) [link to paper](https://arxiv.org/pdf/2209.02341.pdf)
- PETALS: Collaborative Inference and Fine-tuning of Large Models (NeurIPS'22 Workshop WBRC) [link to paper](https://openreview.net/pdf?id=Ls_NTjgWXZV)
- SpecInfer: Accelerating Generative LLM Serving with Speculative Inference and Token Tree Verification (preprint'23) [link to paper](https://www.cs.cmu.edu/~zhihaoj2/papers/specinfer.pdf)
- Fast Distributed Inference Serving for Large Language Models (arXiv'23) [link to paper](https://arxiv.org/pdf/2305.05920.pdf)

## LLM Training Systems

### Single-GPU Systems

- CRAMMING: Training a Language Model on a Single GPU in One Day (arXiv'22) [link to paper](https://arxiv.org/pdf/2212.14034)
- Easy and Efficient Transformer : Scalable Inference Solution For large NLP model (arXiv'22) [link to paper](https://arxiv.org/pdf/2104.12470.pdf)
- High-throughput Generative Inference of Large Language Models with a Single GPU (arXiv'23) [link to paper](https://arxiv.org/pdf/2303.06865.pdf)
- ByteTransformer: A High-Performance Transformer Boosted for Variable-Length Inputs (arXiv'23) [link to paper](https://arxiv.org/pdf/2210.03052.pdf)

### Distributed Systems

- ZeRO: Memory optimizations Toward Training Trillion Parameter Models (SC'20) [link to paper](https://ieeexplore.ieee.org/abstract/document/9355301)
- Megatron-lm: Training multi-billion parameter language models using model parallelism (arXiv'20) [link to paper](https://arxiv.org/pdf/1909.08053.pdf)
- PipeTransformer: Automated Elastic Pipelining for Distributed Training of Large-scale Models Algorithm (ICML'21) [link to paper](http://proceedings.mlr.press/v139/he21a/he21a.pdf)
- Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM (SC'21) [link to paper](https://dl.acm.org/doi/pdf/10.1145/3458817.3476209?casa_token=u0SaPFr_xwsAAAAA:UdIVbVvdimqGt7Wxk6ntI-BHzRl8JxqhkFdZbrXcqV509CHkq8FwQviI7Fsiw7na15IyYcYFf098SQ)
- TeraPipe: Token-Level Pipeline Parallelism for Training Large-Scale Language Models (ICML'21) [link to paper](https://danyangzhuo.com/papers/ICML21-TeraPipe.pdf)
- FastMoE: A Fast Mixture-of-Expert Training System (arXiv'21) [link to paper](https://arxiv.org/pdf/2103.13262.pdf)
- Using DeepSpeed and Megatron to Train Megatron-Turing NLG 530B, A Large-Scale Generative Language Model (arXiv'22) [link to paper](https://arxiv.org/pdf/2201.11990.pdf)
- Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning (OSDI'22) [link to paper](https://www.usenix.org/system/files/osdi22-zheng-lianmin.pdf)
- LightSeq2: Accelerated Training for Transformer-Based Models on GPUs (SC'22) [link to paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10046070&casa_token=Y7UD4u5ej2AAAAAA:sxe5BGbxS2cG0l2vGg7f7L_RchYiovUzvTFgwLC5zRI96PtEzqGLt0TjOpLFvQW4jb6_y7J3R6U)
- Pathways: Asynchronous Distributed Dataflow for ML (arXiv'22) [link to paper](https://arxiv.org/pdf/2203.12533.pdf)
- Varuna: Scalable, Low-cost Training of Massive Deep Learning Models (EuroSys'22) [link to paper](https://dl.acm.org/doi/pdf/10.1145/3492321.3519584)
- FasterMoE: modeling and optimizing training of large-scale dynamic pre-trained models (PPoPP'22) [link to paper](https://dl.acm.org/doi/pdf/10.1145/3503221.3508418)
- PanGu-Σ: Towards Trillion Parameter Language Model with Sparse Heterogeneous Computing (arXiv'23) [link to paper](https://arxiv.org/abs/2303.10845)
- Mobius: Fine Tuning Large-Scale Models on Commodity GPU Servers (ASPLOS'23) [link to paper](https://dl.acm.org/doi/10.1145/3575693.3575703)
- Optimus-CC: Efficient Large NLP Model Training with 3D Parallelism Aware Communication Compression（ASPLOS'23) [link to paper](https://dl.acm.org/doi/pdf/10.1145/3575693.3575712)

## General MLSys-Related Techniques (Not Complete)

- Efficient GPU Spatial-Temporal Multitasking (TPDS'14) [link to paper](https://ieeexplore.ieee.org/document/6777559)
- Enabling preemptive multiprogramming on GPUs (ISCA'14) [link to paper](https://ieeexplore.ieee.org/document/6853208)
- Chimera: Collaborative Preemption for Multitasking on a Shared GPU (ASPLOS'15) [link to paper](https://cccp.eecs.umich.edu/papers/jasonjk-asplos15.pdf)
- Simultaneous Multikernel GPU: Multi-tasking Throughput Processors via Fine-Grained Sharing (HPCA'16) [link to paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7446078&casa_token=vxsr7PVfmXgAAAAA:50JiSZDt8Xzg0lr5tIMu6nlyIRpQawD4HVePmPI-pBOHylszpzBlwPgLEeAPhOhl6cXrHLGhNrg&tag=1)
- FLEP: Enabling Flexible and Efficient Preemption on GPUs (ASPLOS'17) [link to paper](https://dl.acm.org/doi/10.1145/3037697.3037742)
- Dynamic Resource Management for Efficient Utilization of Multitasking GPUs (ASPLOS'17) [link to paper](https://dl.acm.org/doi/10.1145/3037697.3037707)
- Mesh-TensorFlow: Deep Learning for Supercomputers (NeurIPS'18) [link to paper](https://proceedings.neurips.cc/paper_files/paper/2018/file/3a37abdeefe1dab1b30f7c5c7e581b93-Paper.pdf)
- PipeDream: Fast and Efficient Pipeline Parallel DNN Training (SOSP'19) [link to paper](https://dl.acm.org/doi/10.1145/3341301.3359646)
- GPipe: Easy Scaling with Micro-Batch Pipeline Parallelism (NeurIPS'19) [link to paper](https://proceedings.neurips.cc/paper/2019/file/093f65e080a295f8076b1c5722a46aa2-Paper.pdf)
- PipeSwitch: Fast Pipelined Context Switching for Deep Learning Applications (OSDI'20) [link to paper](https://www.usenix.org/system/files/osdi20-bai.pdf)
- Microsecond-scale Preemption for Concurrent GPU-accelerated DNN Inferences (OSDI'22) [link to paper](https://www.usenix.org/conference/osdi22/presentation/han)
- Overlap Communication with Dependent Computation via Decomposition in Large Deep Learning Models (ASPLOS'23) [link to paper](https://dl.acm.org/doi/pdf/10.1145/3567955.3567959)


## LLM Algorithm Papers Recommended for System Researchers

- Attention is all you need (NeurIPS'17) [link to paper](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
- Language Models are Unsupervised Multitask Learners (preprint from OpenAI) [link to paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- Improving Language Understanding by Generative Pretraining (preprint from OpenAI) [link to paper](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
- Language Models are Few-Shot Learners (NeurIPS'20) [link to paper](https://papers.nips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf)
- Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (JMLR'20) [link to paper](https://jmlr.org/papers/volume21/20-074/20-074.pdf)
- Multitask Prompted Training Enables Zero-Shot Task Generalization (ICLR'22) [link to paper](https://openreview.net/pdf?id=9Vrb9D0WI4)
- Finetuned Language Models are Zero-Shot Learners (ICLR'22) [link to paper](https://openreview.net/forum?id=gEZrGCozdqR)
- GLaM: Efficient Scaling of Language Models with Mixture-of-Experts (ICML'22) [link to paper](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C15&q=GLaM%3A+Efficient+Scaling+of+Language+Models+with+Mixture-of-Experts&btnG=)
- LaMDA: Language Models for Dialog Applications (arXiv'22) [link to paper](https://arxiv.org/pdf/2201.08239.pdf)
- PaLM: Scaling Language Modeling with Pathways (arXiv'22) [link to paper](https://arxiv.org/pdf/2204.02311.pdf)
- OPT: Open Pre-trained Transformer Language Models (arXiv'22) [link to paper](https://arxiv.org/pdf/2205.01068.pdf?fbclid=IwAR1_0YiQKgxIsy8unzoLvL9E2OA41_kze-H0YvhoCzIQUp_gk-MR9dUs2ZE)
- Holistic Evaluation of Language Models (arXiv'22) [link to paper](https://arxiv.org/pdf/2211.09110.pdf)
- BLOOM: A 176B-Parameter Open-Access Multilingual Language Model (arXiv'23) [link to paper](https://arxiv.org/pdf/2211.05100.pdf)
- LLaMA: Open and Efficient Foundation Language Models (arXiv'23) [link to paper](https://scontent-atl3-1.xx.fbcdn.net/v/t39.8562-6/333078981_693988129081760_4712707815225756708_n.pdf?_nc_cat=108&ccb=1-7&_nc_sid=ad8a9d&_nc_ohc=fskqjIsP1vwAX-oLQNg&_nc_ht=scontent-atl3-1.xx&oh=00_AfBWZMuYRZYFd8oGUIxSdjKcG-EhmQodKMs7-M_IyuYlPw&oe=64216DE2)
- DeepMind: Training Compute Optimal Large Language Models (preprint from DeepMind) [link to paper](https://arxiv.org/pdf/2203.15556.pdf)
- Scaling Laws for Neural Language Models (preprint) [link to paper](https://arxiv.org/pdf/2001.08361.pdf)
- Scaling Language Models: Methods, Analysis & Insights from Training Gopher (preprint from DeepMind) [link to paper](https://arxiv.org/pdf/2112.11446.pdf)
- LLM-Adapters: An Adapter Family for Parameter-Efficient Fine-Tuning of Large Language Models (arXiv'23) [link to paper](https://arxiv.org/pdf/2304.01933.pdf)

## Survyes
- A Survey of Large Language Models (arXiv'23) [link to paper](https://arxiv.org/abs/2303.18223)


## Other Useful Resources

- [FasterTransformer](https://on-demand.gputechconf.com/gtc-cn/2019/pdf/CN9468/presentation.pdf)
- [An Awesome ML Course](https://www.youtube.com/@deeplearningsystemscourse1116)
