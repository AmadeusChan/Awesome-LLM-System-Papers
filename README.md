# Awesome-LLM-Systems-Papers

## LLM Algorithm Papers Recommended for System Researchers

- Attention is all you need (NeurIPS'17) [link to paper](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
- Language Models are Unsupervised Multitask Learners (preprint from OpenAI) [link to paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- Improving Language Understanding by Generative Pretraining (preprint from OpenAI) [link to paper](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
- Language Models are Few-Shot Learners (NeurIPS'20) [link to paper](https://papers.nips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf)

- DeepMind: Training Compute Optimal Large Language Models (preprint from DeepMind) [link to paper](https://arxiv.org/pdf/2203.15556.pdf)
- Scaling Laws for Neural Language Models (preprint) [link to paper](https://arxiv.org/pdf/2001.08361.pdf)
- Scaling Language Models: Methods, Analysis & Insights from Training Gopher (preprint from DeepMind) [link to paper](https://arxiv.org/pdf/2112.11446.pdf)

## Algorithm-System Co-Design

- DeepSpeed-MOE: Advancing Mixture of Experts Inference and Training to Power Next-Generation AI Scale (ICML'22) [link to paper](https://proceedings.mlr.press/v162/rajbhandari22a/rajbhandari22a.pdf)
- Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity (JMLR'21) [link to paper](https://www.jmlr.org/papers/volume23/21-0998/21-0998.pdf)
- Scalable and Efficient MoE Training for Multitask Multilingual Models (preprint) [link to paper](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Scalable+and+Efficient+MoE+Training+for+Multitask+Multilingual+Models&btnG=)

## LLM Inference (Serving) Systems

### Single-GPU Systems

- TurboTransformers: An Efficient GPU Serving System For Transformer Models (PPoPP'21) [link to paper](https://dl.acm.org/doi/pdf/10.1145/3437801.3441578)
- PetS: A Unified Framework for Parameter-Efficient Transformers Serving (ATC'22) [link to paper](https://www.usenix.org/system/files/atc22-zhou-zhe.pdf)

### Distributed Systems

- Orca: A Distributed Serving System for Transformer-Based Generative Models (OSDI'22) [link to paper](https://www.usenix.org/conference/osdi22/presentation/yu)
- DeepSpeed-inference: enabling efficient inference of transformer models at unprecedented scale (SC'22) [link to paper](https://dl.acm.org/doi/abs/10.5555/3571885.3571946)
- EnergeonAI: An Inference System for 10-100 Billion Parameter Transformer Models (arXiv'22) [link to paper](https://arxiv.org/pdf/2209.02341.pdf)
- PETALS: Collaborative Inference and Fine-tuning of Large Models (NeurIPS'22 Workshop WBRC) [link to paper](https://openreview.net/pdf?id=Ls_NTjgWXZV)

## LLM Training Systems

### Single-GPU Systems

- CRAMMING: Training a Language Model on a Single GPU in One Day (arXiv'22) [link to paper](https://arxiv.org/pdf/2212.14034)
- Easy and Efficient Transformer : Scalable Inference Solution For large NLP model (arXiv'22) [link to paper](https://arxiv.org/pdf/2104.12470.pdf)
- High-throughput Generative Inference of Large Language Models with a Single GPU (arXiv'23) [link to paper](https://arxiv.org/pdf/2303.06865.pdf)

### Distributed Systems

- ZeRO: Memory optimizations Toward Training Trillion Parameter Models (SC'20) [link to paper](https://ieeexplore.ieee.org/abstract/document/9355301)
- Megatron-lm: Training multi-billion parameter language models using model parallelism (arXiv) [link to paper](https://arxiv.org/pdf/1909.08053.pdf)
- PipeTransformer: Automated Elastic Pipelining for Distributed Training of Large-scale Models Algorithm (ICML'21) [link to paper](http://proceedings.mlr.press/v139/he21a/he21a.pdf)
- Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM (SC'21) [link to paper](https://dl.acm.org/doi/pdf/10.1145/3458817.3476209?casa_token=u0SaPFr_xwsAAAAA:UdIVbVvdimqGt7Wxk6ntI-BHzRl8JxqhkFdZbrXcqV509CHkq8FwQviI7Fsiw7na15IyYcYFf098SQ)
- Using DeepSpeed and Megatron to Train Megatron-Turing NLG 530B, A Large-Scale Generative Language Model (arXiv'22) [link to paper](https://arxiv.org/pdf/2201.11990.pdf)

## General MLSys-Related Techniques (Not Complete)

- Efficient GPU Spatial-Temporal Multitasking (TPDS'14) [link to paper](https://ieeexplore.ieee.org/document/6777559)
- Enabling preemptive multiprogramming on GPUs (ISCA'14) [link to paper](https://ieeexplore.ieee.org/document/6853208)
- Chimera: Collaborative Preemption for Multitasking on a Shared GPU (ASPLOS'15) [link to paper](https://cccp.eecs.umich.edu/papers/jasonjk-asplos15.pdf)
- Simultaneous Multikernel GPU: Multi-tasking Throughput Processors via Fine-Grained Sharing (HPCA'16) [link to paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7446078&casa_token=vxsr7PVfmXgAAAAA:50JiSZDt8Xzg0lr5tIMu6nlyIRpQawD4HVePmPI-pBOHylszpzBlwPgLEeAPhOhl6cXrHLGhNrg&tag=1)
- FLEP: Enabling Flexible and Efficient Preemption on GPUs (ASPLOS'17) [link to paper](https://dl.acm.org/doi/10.1145/3037697.3037742)
- Dynamic Resource Management for Efficient Utilization of Multitasking GPUs (ASPLOS'17) [link to paper](https://dl.acm.org/doi/10.1145/3037697.3037707)
- PipeDream: Fast and Efficient Pipeline Parallel DNN Training (SOSP'19) [link to paper](https://dl.acm.org/doi/10.1145/3341301.3359646)
- GPipe: Easy Scaling with Micro-Batch Pipeline Parallelism (NeurIPS'19) [link to paper](https://proceedings.neurips.cc/paper/2019/file/093f65e080a295f8076b1c5722a46aa2-Paper.pdf)
- Microsecond-scale Preemption for Concurrent GPU-accelerated DNN Inferences (OSDI'22) [link to paper](https://www.usenix.org/conference/osdi22/presentation/han)

## Other Useful Resources

- [FasterTransformer](https://on-demand.gputechconf.com/gtc-cn/2019/pdf/CN9468/presentation.pdf)
