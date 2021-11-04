
### Language models are Few Shot learners 

--> Trend towards pre-trained language representations in NLP systems, applied in increasingly flexible and task-agnostic ways for downstream transfer
		1. single representation using word vectors
		2. RNN with multi layer representations and context state
		3. pre-trained recurrent or transformer models, directly fine-tuned, removing the need for task specific architectures
while no task-specific architecture, still needs task-specific fine-tuning and datasets --> problems of specific data collection, a large model then pre-trained on a specific task may infer poorly from unseen data, can't match human's ability to perform a wide variety of language tasks based on a simple instruction or example

--> meta-learning – the model develops a broad set of skills and pattern recognition abilities at training time, and then uses those abilities at inference time to rapidly adapt to or recognize the desired task

   . results still inferior as fine tuning

   . models needs improvement to make meta learning a viable approach, transformer language models from 100 million to 17 billion of parameters in recent years

![image](https://user-images.githubusercontent.com/89974426/140300401-fc55c2ec-40d9-4f8f-a567-1315cbdf03a1.png)


--> GPT-3 – 175 billion parameter autoregressive language model – for each task, we evaluate GPT-3 under 3 conditions: (a) “few-shot learning”, or in-context learning where we allow as many demonstrations as will fit into the model’s context window (typically 10 to 100), (b) “one-shot learning”, where we allow only one demonstration, and (c) “zero-shot” learning, where no demonstrations are allowed and only an instruction in natural language is given to the model. GPT-3 could also in principle be evaluated in the traditional fine-tuning setting, but we leave this to future work
different settings (approaches) for model training

**(1) Fine-Tuning (FT)** has been the most common approach in recent years, and involves updating the weights of a pre-trained model by training on a supervised dataset specific to the desired task. Typically thousands to hundreds of thousands of labeled examples are used. The main advantage of fine-tuning is strong performance on many benchmarks. The main disadvantages are the need for a new large dataset for every task, the potential for poor generalization out-of-distribution [MPL19], and the potential to exploit spurious features of the training data, potentially resulting in an unfair comparison with human performance. 

**(2) Few-Shot (FS)** - model is given a few demonstrations of the task at inference time as conditioning, but no weight updates are allowed. 	

*Advantages*

.need for little task-specific data 

.reduced potential to learn an overly narrow distribution from a large but narrow fine-tuning dataset
disadvantages 

.much worse than state-of-the-art fine-tuned models so far

.a small amount of task specific data is still required

**(3)One-Shot (1S)**- one example given which mimics one way humans can be given language related tasks

**(4)Zero-Shot (0S)** – no example is allowed, only an instruction describing the task, which is the most frequent way human are given tasks

*Training process*

Larger models to use larger batch size but smaller learning rate

*Results*

Assessed on various language tasks

.language modeling tasks and similar tasks, such as Cloze tasks and sentence/paragraph completion tasks

.“closed book” question answering tasks: tasks which require using the information stored in the model’s parameters to answer general knowledge questions

.translation between languages  

.Winograd Schema-like tasks

.tasks involving commonsense reasoning and question answering

.reading comprehension tasks

.SuperGLUE benchmark suite

.NLI

.additional tasks designed especially to probe in-context learning abilities – on-the-fly reasoning, adaptation skills, or open-ended text synthesis

--> Perplexity per word, assessing a language model, which is a probability distribution over sentences or text. An intrinsic measure, independent of the language task. 

N-gram language models are making predictions based on probabilities conditional to the last n tokens, a hidden markov chain where conditional probability of next word in a sentence is described by

![image](https://user-images.githubusercontent.com/89974426/140301312-0355c543-0371-4910-aeb2-15ef0e597df2.png)







