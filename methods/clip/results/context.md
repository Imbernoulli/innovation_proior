# Context

## Research question

Can a visual model be trained so that the **set of things it can recognize is not fixed in advance**, and so that **what it should predict can be specified at test time using ordinary language**, without collecting new labeled data?

The prevailing recipe for state-of-the-art vision is to pretrain a backbone to predict a fixed, predetermined set of object categories — 1000 classes for ImageNet, ~18k for the larger noisily-labeled collections. This has two structural costs. First, supervision is bounded by the label vocabulary: every new visual concept a practitioner cares about requires a fresh batch of crowd-labeled examples in the canonical "1-of-N gold label" format, so scaling supervision means scaling annotation budgets. Second, prediction goes through a static softmax head wired to those exact classes; there is no mechanism for producing an output the model was not explicitly trained to produce, which sharply limits any kind of transfer to a new task without retraining a new head on new data.

By contrast, language pretraining had, by this point, largely escaped both costs. Task-agnostic objectives (next-token and masked-token prediction) consume raw web text with no human labels and improve steadily across many orders of magnitude of compute and data; and once everything is cast as text-to-text, a single pretrained model transfers zero-shot to new tasks by being *told* the task in words, with no task-specific output head. The open question is whether the same task-agnostic, language-specified, zero-shot behavior can be obtained for perception.

A satisfactory solution would have to: draw its supervision from the open vocabulary of natural language rather than a closed label set; be cheap enough per example to absorb hundreds of millions of image–text pairs from the web; and yield a model whose classifier can be *constructed from a text description at inference time*, so new categories cost a sentence rather than a labeled dataset.

## Background

**The field state.** In language, task-agnostic pretraining on web text had become dominant and was demonstrably scalable; flagship autoregressive language models were competitive with bespoke systems across many tasks while needing little or no task-specific data, and zero-shot transfer was treated as a measurement of the *task-learning* a model had acquired during pretraining (Radford et al. 2018, 2019; Brown et al. 2020). In vision, the same period saw large weakly-supervised pretraining push accuracy upward — predicting Instagram hashtags on billions of images (Mahajan et al. 2018) or noisy labels of a 300M-image dataset (Kolesnikov et al. 2019; Dosovitskiy et al. 2020) — but always through a *fixed* label set and a static softmax.

**Learning perception from paired text is an old idea.** Mori et al. (1999) trained a model to predict the nouns and adjectives in documents paired with images to improve retrieval; Quattoni et al. (2007) learned more data-efficient representations from classifiers trained to predict caption words; Srivastava & Salakhutdinov (2012) trained multimodal Deep Boltzmann Machines on image and text-tag features. Joulin et al. (2016) modernized the line: a convolutional network trained to predict the **bag of words** of an image's title/description/hashtag metadata learned representations competitive with supervised ImageNet pretraining on transfer tasks. Li et al. (2017, "Visual N-Grams") extended prediction to phrase n-grams and were the first to attempt **zero-shot transfer to standard image-classification datasets** by scoring each dataset's class names under their model and predicting the highest. More recent proofs of concept used transformer-based caption generation, masked language modeling, and contrastive objectives to learn visual features from text (Desai & Johnson 2020, "VirTex"; ICMLM; Zhang et al. 2020, "ConVIRT").

**Why this line had stayed marginal — the diagnostic facts.** Demonstrated performance was far below the alternatives: Visual N-Grams reached only **11.5%** ImageNet accuracy zero-shot, against ~50% for classic computer vision and ~88% for the contemporaneous supervised state of the art. A crucial difference was scale: the weakly-supervised hashtag/JFT efforts trained for accelerator-*years* on millions-to-billions of images, whereas the text-supervised proofs of concept trained for accelerator-*days* on one to two hundred thousand images. So two things were simultaneously true: natural language is a far broader supervision signal than any fixed label set, yet nobody had run it at the scale where NLP's breakthroughs appeared.

**Contrastive vs. predictive objectives.** A separate strand established that **contrastive objectives can learn better representations than the equivalent predictive objective** (Tian et al. 2019, "Contrastive Multiview Coding"), and that **generative models of images, while capable, need over an order of magnitude more compute than contrastive models for the same representation quality** (Chen et al. 2020, "Generative Pretraining from Pixels"). The contrastive machinery itself had a clear lineage: the batch-as-classification objective with one positive and many in-batch negatives appeared as the **multi-class N-pair loss** (Sohn 2016) and was popularized for representation learning as **InfoNCE**, a lower bound on mutual information (van den Oord et al. 2018, "CPC"). Instance-discrimination work used a non-parametric softmax with a **temperature** of 0.07 over normalized features (Wu et al. 2018). SimCLR-style methods (Bachman et al. 2019; Chen et al. 2020) introduced a **non-linear projection head** between the representation and the contrastive embedding space.

**Generating a classifier from language.** A small literature framed a classifier's *weights* as something that can be produced from a text description: "Write a classifier" generated a zero-shot classifier from purely textual descriptions (Elhoseiny et al. 2013), and Lei Ba et al. (2015) predicted the parameters of a deep zero-shot classifier from text. The general device of a network that emits the weights of another network is the hypernetwork (Ha et al. 2016). Separately, in NLP, task-learning was observed to emerge as a side effect of scaled generative pretraining and to be measurable via zero-shot transfer (Larochelle et al. 2008 "zero-data learning"; Liu et al. 2018; Radford et al. 2018, 2019).

## Baselines

**Caption generation (VirTex, Desai & Johnson 2020).** Jointly train an image CNN and a text transformer from scratch to *generate* the image's caption with an autoregressive language-modeling loss over the exact token sequence. Demonstrated that transformer caption modeling yields useful visual features. Gap: trained on small high-quality caption sets (~100k images, MSCOCO); generating exact words is an expensive objective that spends capacity on phrasing.

**Bag-of-words prediction (Joulin et al. 2016).** Train a CNN to predict the *set* of words occurring in an image's metadata as a multi-label classification target — order-free, no decoder. Competitive with supervised pretraining on transfer. Gap: still tries to predict which specific words appear; the prediction layer is a fixed multi-label vocabulary, and it has no built-in way to be re-pointed at arbitrary new classes at test time.

**Visual N-Grams (Li et al. 2017).** Learn parameters for a dictionary of ~143k visual n-grams (1- to 5-grams) with a differential Jelinek-Mercer smoothing objective; perform zero-shot transfer by converting each class name to its n-gram representation and scoring it. The first system to do zero-shot transfer to standard classification datasets with a generically pretrained model. Gap: 11.5% ImageNet zero-shot — a proof of concept, an order of magnitude below usable accuracy; pre-transformer; small data.

**Contrastive image–text on a narrow domain (ConVIRT, Zhang et al. 2020).** Train paired image and text encoders with a contrastive objective on medical (image, text) pairs, using non-linear projection heads, a function that samples a single sentence from the text, image augmentation, and initialization from pretrained weights. Establishes that the contrastive matching objective transfers across modalities. Gap: applied at small scale in a single specialized domain; carries several components (sentence sampling, non-linear heads, pretrained init) whose necessity at large general-domain scale is untested.

**Weakly-supervised fixed-label pretraining (Mahajan et al. 2018; Kolesnikov et al. 2019, "BiT"; Dosovitskiy et al. 2020, "ViT").** Pretrain on Instagram-hashtag prediction (up to 3.5B images) or noisy JFT-300M labels, then transfer. Large, accurate, the pragmatic middle ground between gold labels and raw text. Gap: supervision is deliberately *designed and limited* to 1000 / 18291 classes, and prediction is a static softmax — so the approach has no native zero-shot mechanism and inherits exactly the fixed-vocabulary ceiling we want to break.

## Evaluation settings

The natural yardstick is transfer to a broad battery of existing classification datasets, measured two ways. **Zero-shot transfer**: no examples from the target dataset are used; the model classifies using only the dataset's class names, mapping the integer label ids back to their English names. **Linear-probe representation evaluation**: freeze the model, fit a regularized logistic-regression classifier (e.g. scikit-learn L-BFGS, with an L2-strength sweep) on features from the penultimate layer, and report the dataset's metric. A standardized 12-dataset suite (Kornblith et al. 2019) plus a broader collection spanning general object recognition (ImageNet, CIFAR-10/100, STL-10, Caltech-101, Pascal VOC), fine-grained recognition (Food-101, Stanford Cars, FGVC Aircraft, Oxford Flowers/Pets, Birdsnap), scene and texture (SUN397, DTD), OCR and rendered text (SVHN, MNIST, IIIT5K, rendered sentiment text), satellite imagery (EuroSAT, RESISC45), action recognition in video (UCF101, Kinetics-700, using a center frame), counting (CLEVR), traffic signs (GTSRB), medical (PatchCamelyon), and geo-localization. Metrics are per-dataset: top-1 accuracy, mean-per-class accuracy, 11-point mAP, ROC-AUC. Robustness to natural distribution shift is assessed on ImageNet-derived shift sets (ImageNetV2, ImageNet-Sketch, ImageNet-A, ImageNet-R, ObjectNet, and video-derived sets), comparing accuracy on shifted distributions to in-distribution accuracy. Image–text retrieval (Flickr30k, MSCOCO; Recall@K) directly probes whether learned representations preserve paired image/text associations.

## Code framework

A minimal dual-encoder training harness starts from a paired `(image, text)` dataloader, an image backbone, a transformer text encoder over byte-pair tokens, an optimizer, and a training loop. What remains open: how the two encoded streams are coupled into a training signal, and how a trained model would be re-pointed at an unseen downstream task at inference time.

```python
import torch.nn as nn

image_encoder = build_image_encoder()   # CNN or ViT -> [n, d_i]
text_encoder  = build_text_encoder()    # transformer over BPE tokens -> [n, d_t]


class DualEncoderModel(nn.Module):
    def __init__(self, image_encoder, text_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder  = text_encoder
        # TODO: any adapters / scalar parameters the coupling objective needs

    def encode_image(self, image):
        # TODO: map an image to the representation used for the objective
        pass

    def encode_text(self, text):
        # TODO: map a piece of text to the representation used for the objective
        pass

    def forward(self, image, text):
        return self.encode_image(image), self.encode_text(text)


def training_objective(image_features, text_features, model):
    # TODO: how should a batch of matched (image, text) representations
    #       supervise the two encoders against each other?
    pass


def build_classifier(model, task_specification):
    # TODO: turn a description of an unseen task into a classifier with no labeled data
    pass


def classify(model, image, classifier):
    # TODO: score the image against the classifier
    pass


def train(model, dataloader, optimizer):
    for image, text in dataloader:
        image_features, text_features = model(image, text)
        loss = training_objective(image_features, text_features, model)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
```
