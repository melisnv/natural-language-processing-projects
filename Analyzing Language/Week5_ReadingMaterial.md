# Error Analysis: Bias and Interpretability(Yorumlanabilirlik)

`Bias` is a preset/prior that can help us make decisions in absence of more information. <br>
`Bender Rule:` Always name the language(s) you are working.<br>

## Bias in NLP

Problematic when a system **only** relies on biases and not evidence. Acknowledgement and awareness of biases is extremely important. Where does bias come from ? There can be different sources of bias such as data, annotations, representations, models, or research design. There are 2 types of biases when collecting data : Reporting bias and Selection bias. `Reporting Bias` is what people share is not a reflection of real-world frequencies. **Reporting Bias** (_raporlama yanlılığı_), denekler tarafından "bilginin seçici olarak açıklanması veya bastırılması" olarak tanımlanır. Yapay zeka araştırmalarında, raporlama yanlılığı terimi, insanların mevcut tüm bilgileri yetersiz rapor etme eğilimine atıfta bulunmak için kullanılır._Reporting bias means that only a selection of results are included in any analysis, which typically covers only a fraction of relevant evidence._
`Selection Bias` does not reflect a random sample. **Selection Bias** (_seçim yanlılığı_), analiz edilen kişilerin, grupların ya da verinin gerekli rastgeleliği sağlayamayacak şekilde seçildiği ve dolayısıyla elde edilen örneklemin incelenmek istenen popülasyonu temsil etmekte yetersiz kaldığı yanlılıktır. These biases were from data collection part. `Annotator Bias` is a form of bias that annotators cause due to different knowledge in regards to the task and their subjective perception.

`Input Representation Bias`

- Word embeddings<br>
- Pre-trained language models<br>
  These representations are trained on loads of data which is mostly not selected carefully.

#### Should such biases actually be there in our model?

If we want to replace/integrate such models into our (daily) lives, we have to think about _fairness_, _accountability_, _transparency_.

## Interpretability

`Interpretability` is the ability to explain or to present in understandable terms to a human.
The area of interpretability is very important because we can not state whether our model is good by looking at the weights. Therefore, we do not know what's happening inside of the box(model).`Intrinsic` model is already interpretable by itself. With huge models, we need techniques that are not internal to the model itself. And these techniques are called `Post-hoc` techniques where we add to our model in order to make it more _explainable_.(could added after the training)
_Bilimsel bir çalışmada, `post hoc` analiz, veriler görüldükten sonra belirtilen istatistiksel analizlerden oluşmaktadır. _

`Plausibility` refers to how convincing the interpretation is to humans, while `faithfulness` refers to how accurately it reflects the true reasoning process of the model. They can be both `Plausible` but only one of them can be `Faitfull`.

In a recent paper on _explainability_, we refer to **global interpretability** _(explanations on the whole model behavior)_ as a “user's manual” approach. **Local explainability** _(explanations on an input instance)_ refers to the ability of the system to tell a user why a particular decision was made.

![explanations](https://firebasestorage.googleapis.com/v0/b/birthday-react-6eca4.appspot.com/o/Natural-Language-Processing%2FWeek%205%20%3A%20Bias%20and%20Interpretability%2Fglocal.jpg?alt=media&token=d3fa42a5-b1fc-4fa0-be06-a1bf95dbce96)

## Types of Interpretability

![Types of Interpretability](https://firebasestorage.googleapis.com/v0/b/birthday-react-6eca4.appspot.com/o/Natural-Language-Processing%2FWeek%205%20%3A%20Bias%20and%20Interpretability%2Finter.jpg?alt=media&token=dad0388b-08c7-48d6-9035-8cc3d84ef388)

`Sample Similarity` is you look similar samples as the example you want to explanation for. By looking at the samples that are similar to training data that your model has trained off. Given an input instance, examples in the training data are found that are similar to the input example. Can help in understanding why a certain label was given and identifying mistakes in the dataset.<br>
_"I really like ice cream"_ -> negative <br>
**Similar samples:**<br>
_”Mcdonald’s ice cream is something I really like”_ -> negative (0.90)<br>
_”ice cream is bad for you”_ -> negative(0.8)<br>

Simple way to find these sample of similarities is to use `cosine similarity`. The technique `Influence functions` calculates loss of the model on the input instance by removing an example in the training dataset and see if there is a change in the loss of model. If loss is small that means it wasn't that influencial however this technique is very expensive and takes a lot of time which is a probem as scability. Found to be faithful (not much work done in NLP) but expensive to compute.<br>

`Input Features` is highlighting tokens of the input that are used for the prediction. We can use _attention weights_ to _highlight_ those _tokens_.

![](https://firebasestorage.googleapis.com/v0/b/birthday-react-6eca4.appspot.com/o/Natural-Language-Processing%2FWeek%205%20%3A%20Bias%20and%20Interpretability%2Fwth.jpg?alt=media&token=a0ea6bc5-a269-4829-b03d-679d0cebccbb)

## Input Modification

_Input modification_ is a way to change things in your input and based on the changes we'll have an idea like _"This explains the absence of the token"_ or _"The addition of this token is the reason why the model changes."_ `Counterfactuals` _(karşıolgusal)_: How can we change the input such that the
prediction changes?

![](https://firebasestorage.googleapis.com/v0/b/birthday-react-6eca4.appspot.com/o/Natural-Language-Processing%2FWeek%205%20%3A%20Bias%20and%20Interpretability%2Fwht3.jpg?alt=media&token=93055e81-93f0-4c6a-9a72-295f49dcb35f)

There are issues with its faithfulness, sometimes these _changes are not directed by model behavior_.<br>

`Adversarials` are examples where the model makes an incorrect prediction.<br>
_"I really like ice cream"_ --> positive<br>
**Adversarial:**<br>
_"I really like **yuuuuuummmy** ice cream"_ --> negative<br>
_"I **rlly** like ice cream"_ --> negative<br>

Adversarials are hard to find a direction.

## Probing

We use the emmbedings and these representations that our in our pre-trained or fine-tuned model. Then we use those representation to predict if the model has a linguistic property is encoded in them or not. By using probing we can have more of an idea of our model on a **global** level like _what kind of linguistic information has been captured by our model_. Global explanation use the representations of a pre-trained or fine-tuned model to predict a linguistic property of an input sample. Not much work done on faithfulness.<br>

### Why is interpretability important?

**Accountability**: When a model fails, why is that the case?<br>
**Ethics**: Is the model using biased decision-making, e.g. racism, sexism?<br>
**Safety**: Can we trust the model in the real world?<br>
**Scientific Understanding**: Generate hypotheses, model debugging, why is my model failing on such an instance?<br>

## Explain the different types of interpretability.

The difference between **local** and **global**. Probing is global but rest is local. Sample similarity relates to data. Probing relates to my model. Input features that input modifications that are where I can play around my input instance and see the effect on output.
