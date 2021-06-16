# prior_knowledge_matrix_for_sequence_tagging

**Paper name (useful phrases)**:
* Bring prior knowledge 
* Constraints in sequence tagging (maybe not only)
* Use it knowledge in training process
* Add it to loss
* Prior knowledge matrix
* Faster convergence
* Unsupervised (we don't need labels to compute gumbel loss)


**Data**:
* [Datasets for Entity Recognition](https://github.com/juand-r/entity-recognition-datasets) - use this!
* [Annotated Corpus for Named Entity Recognition](https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus/kernels)


**Links for relevant papers, articles, implementations**:
* [Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/abs/1611.01144)
* [The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables](https://arxiv.org/abs/1611.00712)
* [Neural Networks gone wild! They can sample from discrete distributions now!](https://anotherdatum.com/gumbel-gan.html)
* [ ] [Anticipation-RNN: enforcing unary constraints in sequence generation, with application to interactive music generation](https://link.springer.com/article/10.1007/s00521-018-3868-4)
* [ ] [Enhancing Neural Sequence Labeling with Position-Aware Self-Attention](https://arxiv.org/pdf/1908.09128.pdf)
* [ ] [Inference constraints (not in training) in allennlp](https://github.com/allenai/allennlp/blob/master/allennlp/modules/conditional_random_field.py)
* [ ] [Prior initialization for impossible transitions (-10000)](https://github.com/threelittlemonkeys/lstm-crf-pytorch/blob/master/model.py)


**Useful links**:
* [Attention? Attention!](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html) - Lilian Weng Blog
* [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Jay Alammar Blog
* [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) - Harvard Blog
* [15 Free Datasets and Corpora for Named Entity Recognition (NER)](https://lionbridge.ai/datasets/15-free-datasets-and-corpora-for-named-entity-recognition-ner/)
* [PyTorch-Tutorial-to-Sequence-Labeling](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Sequence-Labeling)
* [Sequence tagging example](http://www.cse.chalmers.se/~richajo/nlp2019/l6/Sequence%20tagging%20example.html)
* NER: [tutorial](https://cs230.stanford.edu/blog/namedentity/) - [github](https://github.com/cs230-stanford/cs230-code-examples)
* [Approaching a Named Entity Recognition (NER) — End to End Steps](https://mc.ai/approaching-a-named-entity-recognition-ner%E2%80%8A-%E2%80%8Aend-to-end-steps/)
* [Named Entity Recognition on CoNLL dataset using BiLSTM+CRF implemented with Pytorch](https://pythonawesome.com/named-entity-recognition-on-conll-dataset-using-bilstm-crf-implemented-with-pytorch/)
* [Named Entity Recognition with BiLSTM-CNNs](https://medium.com/illuin/named-entity-recognition-with-bilstm-cnns-632ba83d3d41)
* [How does pytorch backprop through argmax?](https://stackoverflow.com/questions/54969646/how-does-pytorch-backprop-through-argmax)
* [Differentiable Argmax!](https://lucehe.github.io/differentiable-argmax/)
* [Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation](https://arxiv.org/pdf/1308.3432.pdf)
* [nn.Embedding source code](https://pytorch.org/docs/stable/_modules/torch/nn/modules/sparse.html#Embedding)
* [What is the correct way to use OHE lookup table for a pytorch RNN?](https://stackoverflow.com/questions/57632084/what-is-the-correct-way-to-use-ohe-lookup-table-for-a-pytorch-rnn)
* Beam Search: [Andrew Ng YouTube](https://youtu.be/RLWuzLLSIgw) - [How to Implement a Beam Search Decoder for Natural Language Processing](https://machinelearningmastery.com/beam-search-decoder-natural-language-processing/)
* [Backpropgating error to emedding matrix](https://datascience.stackexchange.com/questions/33041/backpropgating-error-to-emedding-matrix)
* [Inside–outside–beginning (tagging)](https://en.wikipedia.org/wiki/Inside–outside–beginning_(tagging))
* [Gumbel-Softmax trick vs Softmax with temperature](https://datascience.stackexchange.com/questions/58376/gumbel-softmax-trick-vs-softmax-with-temperature)
* PyTorchLightning: [github](https://github.com/PyTorchLightning/pytorch-lightning) - [docs](https://pytorch-lightning.readthedocs.io/en/stable/) - [medium](https://towardsdatascience.com/from-pytorch-to-pytorch-lightning-a-gentle-introduction-b371b7caaf09)
* [PyTorch Lightning vs PyTorch Ignite vs Fast.ai](https://towardsdatascience.com/pytorch-lightning-vs-pytorch-ignite-vs-fast-ai-61dc7480ad8a)


**Architecture**:
* RNN with attention
* BERT and friends


**Hypotheses**:
* [ ] Does *gumbel-softmax* take log probabilities as input?
* [ ] Basic configuration for *two-tokens constraints* (matrix), the *three-tokens constraints* and more (tensor), compare convergence and results (although NN could capture difficult dependencies from data, it still strangle to capture temporal dependencies over tokens, especially from long sequence patterns)
* [ ] Check filtering PAD token in gumble loss term
* [ ] Play with parameter *"hard"* in *PyTorch F.gumbel_softmax*
* [ ] Play with *Beam search* (see links)
* [ ] Check *FP/FN* when using *gumbel-softmax*
* [ ] Try to use *softmax* probabilities instead of *gumbel-softmax* samples, what strategy to use
* [ ] Try *F.argmax* with backpropagation (see in links)
* [ ] Since loss is a sum of *cross-entropy loss* and *prior-knowledge loss*, compare its order of magnitude and find out how to select *lambda* automatically (ratio of two losses)
* [ ] Play with reverse matrix (replace 0 to 1 and 1 to 0) and understand, how it affects final score
* [ ] Make *nn.Module* for efficient computing and write it in paper (see link about embedding)
* [ ] Use two classifiers (neural networks) - first for *tokenization*, second for *classification* (halve num of classes)
* [ ] Play with knowledge matrix init (with 0/1 or something else)
* [ ] (additional) Try seq2seq architecture for sequence labelling (maybe find relevant paper and add to references)
* [ ] (additional) Try seq2seq constraints (output length matches input length)


**TODO**:
* [ ] Understand, how presubmit arxiv paper, what rules to publish, and other about conferences
* [ ] Find datasets to compare with (NER, POS, etc)
* [ ] Make table on another page (or another board) for hypothesis results
* [ ] Add POS-tags in useful links
* [ ] Decide, what papers to use for structure (use conferences restrictions)
* [ ] Compare our implementation vs cs230-code-examples (on their dataset)
* [ ] (additional) Maybe use docker during experiments and after (final project version)


**Rules**:
* Save all references to use it later in paper
* Make issues for hypothesis
* Use PyTorchLightning
* Use different folder for different hypothesis


**Metrics**:
