# Qiu ImageRAGTurbo Towards One-step Text-to-Image Generation with Retrieval-Augmented Diffusion Models CVPR 2026 paper


---

## Page 1

ImageRAGTurbo: Towards One-step Text-to-Image Generation with
Retrieval-Augmented Diffusion Models
Peijie Qiu1*, Hariharan Ramshankar2, Arnau Ramisa2, Amit Kumar K C2,
Ren´e Vidal2, Vamsi Salaka2, Rahul Bhagat2
1Washington University in St. Louis, 2Amazon Core Search
peijie,qiu@wustl.edu
{dsthari, aramisay, amitkrkc, rvidal, vsalaka, rbhagat}@amazon.com
Abstract
Diffusion models have emerged as the leading approach for
text-to-image generation. However, their iterative sampling
process, which gradually morphs random noise into coher-
ent images, introduces significant latency that limits their
applicability. While recent few-step diffusion models reduce
the number of sampling steps to as few as one to four steps,
they often compromise image quality and prompt alignment,
especially in one-step generation. Additionally, these mod-
els require computationally expensive training procedures.
To address these limitations, we propose ImageRAGTurbo,
a novel approach to efficiently finetune few-step diffusion
models via retrieval augmentation. Given a text prompt, we
retrieve relevant text-image pairs from a database and use
them to condition the generation process. We argue that
such retrieved examples provide rich contextual informa-
tion to the UNet denoiser that helps reduce the number of
denoising steps without compromising image quality. In-
deed, our initial investigations show that using the retrieved
content to edit the denoiser’s latent space (H-space) with-
out additional finetuning already improves prompt fidelity.
To further improve the quality of the generated images, we
augment the UNet denoiser with a trainable adapter in the
H-space, which efficiently blends the retrieved content with
the target prompt using a cross-attention mechanism. Ex-
perimental results on fast text-to-image generation demon-
strate that our approach produces high-fidelity images with-
out compromising latency compared to existing methods.
1. Introduction
Diffusion models [14, 39–41] have demonstrated remark-
able capabilities in generating high-quality images. How-
ever, their iterative sampling process introduces a signifi-
cant computational bottleneck that limits their practical ap-
plications. In particular, standard diffusion models typically
*This work was done during an internship with Amazon
Figure 1. Illustrative examples of ImageRAGTurbo, where the vi-
sual concepts are highlighted by colored boxes. ImageRAGTurbo
outperforms Stable Diffusion Turbo (adversarially distilled Stable
Diffusion without retrieval) in generating accurate visual concepts.
Even with a single step, ImageRAGTurbo performs comparably to
Stable Diffusion with 50 steps. Please zoom in for better quality.
require 25 −50 denoising steps to generate a single im-
age, with each step involving an expensive network func-
tion evaluation of the denoiser. This sequential sampling
process leads to high latency, limiting the use of these dif-
fusion models in real-time and interactive applications.
Classical approaches to improving the efficiency of dif-
fusion models include (1) latent diffusion models that op-
erate in a lower-resolution latent space to gain efficiency in
the spatial dimension [40], and (2) fast ordinary differential
equation (ODE) samplers that directly reduce the number of
sampling steps [27, 28, 51, 59]. However, despite remark-
This CVPR paper is the Open Access version, provided by the Computer Vision Foundation.
Except for this watermark, it is identical to the accepted version;
the final published version of the proceedings is available on IEEE Xplore.
529

---

## Page 2

able progress in developing fast ODE samplers, they still
require more than 10 steps to generate images with satisfac-
tory quality.
Recent approaches to improving the efficiency of diffu-
sion models include model quantization [24, 25], cached
sampling [32, 62], and few-step distilled models [30, 43,
44, 53]. Although the first two families of methods acceler-
ate sampling, they still require a large number of steps. In
contrast, few-step models distill the full sampling trajectory
of the teacher diffusion models to a short trajectory with
just one to four steps. In the extreme one-step case, the dis-
tilled model learns to directly map noise to the target dis-
tribution. However, this poses the significant challenge of
achieving high image and prompt fidelity1 [43, 61]. For ex-
ample, given the prompt “A boat on the water with a light-
house in the background.”, the distilled few-step model may
fail to generate the visual concept of “boat” in the resulting
images (see illustration in Fig. 1). Additionally, distilling
these few-step models from teacher diffusion models typi-
cally requires substantial computational resources, making
them less accessible for easy adoption.
One potential solution to improve prompt fidelity is to
train preference models [21, 55]. However, this requires
human annotated preference datasets. Another potential so-
lution, drawing inspiration from retrieval-augmented gener-
ation (RAG) in large language models (LLMs) [5, 23, 38],
is to adapt RAG to enhance prompt fidelity in image gener-
ation [1, 3, 47, 48, 58]. These models first query a database
for relevant text-image pairs and then condition the diffu-
sion models on the retrieved pairs to enhance prompt fi-
delity. Although prior retrieval-augmented image genera-
tion (RAIG) models have investigated various conditioning
strategies [1, 3, 48] and fine-grained retrieval content using
LLMs [47, 58], RAIG with few-step diffusion models has
not yet been explored. In addition, most previous methods
do not focus on improving efficiency in both training and
inference [1, 3, 48, 58].
In this paper, we propose to use retrieval augmentation to
improve the generation quality of few-step diffusion models
while maintaining efficiency. The key intuition is that in-
jecting semantically relevant retrieved information into the
diffusion model can simplify the task of mapping random
noise to target distributions. Preliminary results on using
the retrieved content to directly edit the latent space of the
diffusion model’s denoiser (H-space) without any finetun-
ing already show improved image and prompt fidelity (see
Fig. 1). However, this simple approach requires an expen-
sive hyperparameter search at inference time to maximize
performance, compromising the latency benefits of few-step
image generation. To mitigate this challenge, we propose
to augment the denoiser’s H-space with a trainable adapter
that blends the retrieved content with the target prompt us-
1We use prompt fidelity and text-to-image alignment interchangeably.
ing a cross-attention mechanism (Sec. 3.3). Experiments
show that the proposed ImageRAGTurbo method produces
high-fidelity images without compromising efficiency.
Our main contributions can be summarized as follows:
• We introduce a novel retrieval-augmented framework for
few-step image generation with diffusion models, which
significantly improves prompt fidelity and image quality
while maintaining fast generation speeds.
• We develop an efficient finetuning approach using a
lightweight adapter network in the H-space, which re-
duces computational requirements compared to tradi-
tional few-step model training methods.
• We conduct extensive experiments demonstrating that
our method outperforms existing few-step approaches in
terms of both generation quality and prompt alignment,
while maintaining comparable inference speed.
The remainder of the paper is organized as follows:
Section 2 reviews relevant literature, Section 3 details our
method including our preliminary investigation in Sec-
tion 3.2 and the proposed H-adapter finetuning in Sec-
tion 3.3. Section 4 presents our experimental validation and
results, and Section 5 concludes the paper.
2. Related work
Few-step Text-to-Image Generation. Standard diffusion
models [13] generate images via an iterative denoising pro-
cess that aims to reverse the forward noising process step
by step. This iterative reverse sampling process occurs se-
quentially and cannot be parallelized to speed it up. Al-
though fast ODE samplers [27, 28, 51, 59] can skip some
steps by enabling a large sampling step size, these models
cannot directly map noise to images like GANs [7, 19]. To
address this challenge, consistency models [30, 53] enforce
the self-consistency of the denoising networks along the
sampling trajectory such that they can map any point in an
ODE trajectory to its starting point. This enables one-step
or few-step generation by directly mapping random noise
to images, or through intermediate steps. However, despite
being theoretically grounded, consistency models distilled
from pretrained diffusion models, such as latent consistency
models [30], typically result in low-quality samples when
using few steps [44]. In contrast, adversarial diffusion dis-
tillation [44] and its variant in latent space [43] show strong
empirical performance in few-step generation. There are
also hybrid methods that combine consistency models and
adversarial training [4, 60]. However, all these methods
share a common flaw: they often struggle to generate faith-
ful images that align well with the target prompts.
Retrieval-Augmented Image Generation. RAG has been
extensively explored in natural language processing [5, 23,
38], but has received comparatively less attention in the con-
text of image generation. An early example is the retrieval-
530

---

## Page 3

𝜏𝜏𝜙𝜙(𝒑𝒑𝒕𝒕𝒕𝒕𝒕𝒕)
Initial latent 𝒛𝒛𝑡𝑡
𝒉𝒉𝑡𝑡
𝑡𝑡𝑡𝑡𝑡𝑡
𝒉𝒉𝑡𝑡
𝑟𝑟𝑟𝑟𝑟𝑟𝑟𝑟
𝜏𝜏𝜙𝜙𝒑𝒑𝒓𝒓𝒓𝒓𝒓𝒓𝒓𝒓
𝑔𝑔(𝒉𝒉𝑡𝑡
𝑟𝑟𝑟𝑟𝑟𝑟𝑟𝑟, 𝒉𝒉𝑡𝑡
𝑡𝑡𝑡𝑡𝑡𝑡)
∆𝒉𝒉
Denoising UNet
Source latent 
𝒛𝒛𝑡𝑡
𝑟𝑟𝑟𝑟𝑟𝑟𝑟𝑟 
Generated latent ො𝒛𝒛0
Frozen
Trainabe
Target latent 𝒛𝒛0
𝒑𝒑𝒕𝒕𝒕𝒕𝒕𝒕: A cat holding a sign 
that says hello world
Databas
e
𝒑𝒑𝒓𝒓𝒓𝒓𝒓𝒓𝒓𝒓: A cat holding a sign
Pre-cached Retrieval Branch
𝜀𝜀
𝒟𝒟
Optimize
ℒ𝑑𝑑𝑑𝑑𝑑𝑑𝑑𝑑𝑑𝑑𝑑𝑑+ 𝛼𝛼ℒ𝑎𝑎𝑎𝑎𝑎𝑎+ 𝛽𝛽𝛽𝑒𝑒𝑒𝑒𝑒𝑒𝑒𝑒𝑒𝑒𝑒𝑒
Generated image ෝ𝒙𝒙
Text encoder 𝜏𝜏𝜙𝜙(ȉ)
Text encoder 𝜏𝜏𝜙𝜙(ȉ)
ℋ-space
skip-connection
Figure 2. Overview of the proposed ImageRAGTurbo framework for efficiently finetuning the few-step diffusion models with retrieval-
augmented generation. The framework involves two main branches: i) a standard denoising branch (highlighted by green), and ii) a
retrieval branch (highlighted by purple). For a target prompt ptgt, it will first be converted to embeddings with a pretrained text encoder
τϕ(·). Then we query a database based on the target text embeddings to obtain the retrieved text-latent embeddings (τϕ(pretr), zretr
t
),
which is then fed into the encoder of the denoiser to obtain the retrieved H-space feature hretr
t
. Finally, the H-space feature hretr
t
in the
retrieved branch is injected into the denoising branch by the proposed H-space adapter to guide the generation. See details in Sec. 3.
augmented diffusion model (RDM) [1], which conditions
the denoiser of a latent diffusion model [40] on CLIP em-
beddings [37] of image neighbors retrieved from an external
database. Similarly, Re-Imagen [3] and its variant [16] ex-
tend the pixel-space training method of Imagen [41] to con-
dition the denoiser on the retrieved text-image pairs. In ad-
dition, RealRAG [31] aims to enhance the retrieval branch
of retrieval-augmented image generation by training a re-
flective retriever that selects retrieved images to comple-
ment the model’s missing knowledge. However, all these
methods are not optimized for few-step image generation.
Recent approaches like ImageRAG [47] and FineRAG [58]
leverage Multimodal LLMs (MLLMs) for dynamic image
retrieval based on text prompts, and use these MLLMs
to assess and progressively improve the generated images
through iterative refinement. However, the repeated calls
to MLLMs in these methods introduce substantial compu-
tational overhead, which defeats the primary goal of fast
generation in this paper.
3. Method
In this work, we propose ImageRAGTurbo, a framework
for boosting the performance of few-step generation mod-
els. Specifically, given a target prompt, ptgt, we retrieve rel-
evant text-to-image pairs (pretr, xretr) from a database. The
retrieved information is then injected into the text-to-image
diffusion models to guide the generation process. Motivated
by the recent finding that the H-space of the UNET denoiser
already contains semantically meaningful representations,
our investigation starts from a training-free direct H-space
injection (Sec. 3.2) and then extends to the proposed effi-
cient H-space adapter tuning (Sec. 3.3). The overview of
the proposed framework is shown in Fig. 2.
3.1. Preliminary
Latent diffusion models.
In this work we focus on la-
tent diffusion models (LDMs) [40] and their popular suc-
cessor stable diffusion models [6, 36], rather than pixel-
space alternatives like Imagen [41], due to their popularity
and state-of-the-art performance for text-to-image genera-
tion. Specifically, LDMs have two stages: first, a pretrained
VAE that transforms images x to their latent representations
z via the encoder E : x 7→z and then back to the image
space via the decoder D : z 7→x; second, a diffusion model
applied to the resulting latent space Z. LDMs for text-to-
image generation henceforth learn to progressively trans-
form a standard Gaussian distribution ϵ ∼N(0, I) along
with a target prompt ptgt into the target latent z0 = E(x)
through a denoiser fθ:
  \ label { eq :parameter iz a tion } \ha t {{\b m {z}}}_0 \!=\! f_\theta (\bm {z}_t, t, \tau _\phi (\p ^{\tgt })), \bm {z}_t = \alpha _t \bm {z}_0 + \sigma _t \bm {\epsilon }, t\in \mathcal {U}(0, 1),\! (1)
where αt and σt control the noise scheduling, and τϕ(·) is a
text encoder (e.g., a pretrained CLIP text encoder [37]).
531

---

## Page 4

Although there are different parameterizations of the de-
noiser [20], we opt for the one that predicts ˆz0 in Eq. (1), as
it is widely used in few-step diffusion models [30, 43, 44,
53]. The denoiser can be trained by optimizing the scoring
matching objective [52, 54]:
  \
m
in _\theta \mathbb {E}_{\bm {x} \sim p(\bm {x}), \bm {\epsilon }\sim \mathcal {N}(\bm {0}, \bm {I}), t\sim \mathcal {U}(0, 1)} \left [ \lambda (t)\| \hat {{\bm {z}}}_0 - \bm {z}_0 \|_2^2 \right ] ,
(2)
where p(x) is the data distribution, and λ(t) is a weighting
function. After training, we can generate the target latent
ˆz0 by solving an SDE/ODE using a variety of solvers [13,
27, 28, 51, 59], starting from a random Gaussian noise ϵ ∼
N(0, I). The resulting latent ˆz0 is then projected back to
the image space via the VAE decoder D to produce the final
generated image ˆx = D(ˆz0).
H-space of the UNet denoiser. We define the H-space of
the denoiser [22], which plays a crucial role in our method,
as the space of feature maps ht corresponding to the deepest
layer of the UNet denoiser (see Fig. 2). High-level concepts
of a generated image, such as image class or object presence
and attributes, are largely determined by H-space features,
which consequently impact the prompt fidelity [46]. In con-
trast, low-level features corresponding to earlier layers of
the denoiser determine the details of a generated image.
3.2. Retrieval-Augmented Direct H-space Injection
Recent explorations have revealed that the denoising UNet
already has a semantically meaningful H-space [22, 34].
In light of this, we first investigate a simple training-free
mechanism to directly inject the retrieved information into
the H-space of a UNet denoiser to demonstrate the effec-
tiveness of the retrieval augmented generation. Specifically,
we first project the retrieved text-image pairs (pretr, xretr)
to their corresponding embeddings (τϕ(pretr), zretr
0 ), where
zretr
0
= E(xretr) (see illustration in Fig. 2). Subsequently, the
retrieved embeddings (τϕ(pretr), zretr
0 ) are fed to the encoder
of the UNet denoiser f enc
θ
to extract the H-space features:
hretr
t
= f enc
θ
(τϕ(pretr), t, zretr
0 ). In practice, we set t = 0, as
the retrieved images are noise-free. Likewise, we can obtain
the target H-space features htgt
t = f enc
θ
(τϕ(ptgt), t, zt). We
then enrich the target H-space features by blending it with
the retrieved H-space features via spherical normalized in-
terpolation:
  \bm 
{
h }_t^{\ b lend 
} = \
frac 
{
\ sin [(1-
 w) \
Omeg
a _t]}{\sin \Omega _t} \bm {h}^{\retr }_t + \frac {\sin [w \Omega _t]}{\sin \Omega _t} \bm {h}^{\tgt }_t, 
(3)
where Ωt = arccos(⟨htgt
t , hretr
t ⟩), and w ∈(0, 1) con-
trols the strength of the blending.
We employ spherical
normalized interpolation rather than linear interpolation,
as geodesic interpolation paths provide smoother semantic
transitions [26], effectively preventing the abrupt changes
typically caused by phase transitions [46]. The blended H-
space features along with the target prompt embeddings are
Figure 3. Performance of direct H-space injection, shown as a
histogram of TIFA scores across various categories.
then fed to the decoder of the UNet denoiser f dec
θ
to produce
the denoised latent
 \hat { \bm
 
{z}}_{t-1} = f_\t
h eta ^{dec}(\tau _\phi (\text {p}^{\tgt }), t, \bm {h}_t^{\retr }) .
Our initial investigation has revealed that the direct H-
space injection without any further tuning yields a modest
improvement in prompt fidelity (see Fig. 3), increasing the
TIFA score [17] from 0.779 to 0.781 with a fixed w = 0.8
for all prompts. Observing that different prompts require
different optimal blending strengths, we determine the op-
timal w∗that maximizes the TIFA score for each prompt
within a set of predefined values {0.1, 0.2, . . . , 0.9} via
brute force search/exhaustive search. This further improves
the TIFA score to 0.816, even surpassing the 50-step sta-
ble diffusion model without retrieval augmentation. How-
ever, determining the optimal blending strength w∗of each
prompt is an ill-posed problem, as the optimal blending de-
pends on latent factors such as retrieval relevance, prompt
semantics, generation difficulty, etc.
A more principled solution to this problem is to edit the
H-space representations along multiple directions and com-
pute the edit strength automatically. For example, recent
work for LLMs [29] constructs a large dictionary of latent
editing directions (e.g., 40,000) and uses sparse coding to
automatically select a small, relevant subset of directions
along with their corresponding strengths. However, solv-
ing a sparse coding problem during inference can largely
increase the inference time. Next, we propose an efficient
method based on training an adapter that blends retrieved
and target H-space features.
3.3. Retrieval-Augmented Efficient H-space Tuning
Although directly injecting content into H-space improves
prompt fidelity, it requires exhaustive search for the optimal
532

---

## Page 5

Figure 4. The architecture of the discriminator used for latent ad-
versarial training, which adds noise to the samples ˆz0 from the
student model and z0 from the teacher model, and differentiate
them.
blending strength, making it impractical for real-world ap-
plications. This motivates us to automatically learn the cor-
relations between the retrieved and target H-space features.
For this purpose, we introduce a trainable adapter gφ(·, ·) to
the H-space of the UNet denoiser while freezing the other
parts of the denoiser. The adapter gφ(·, ·) leverages a cross-
attention mechanism to automatically determine the corre-
lations between the retrieved and target H-space features:
  \begi
n {spli
t } \!\! & 
\qua
d \
q
u a
d g_ \ varp
h i ( \b m {h}^
{ \ t g t } _t, \
b m {h}^{\retr }_t) = \operatorname {softmax}\left (\frac {\bm {Q} \bm {K}^\top }{\sqrt {d_k}} \right ) \bm {V}, \\ & \bm {Q} = \bm {W}_Q \cdot \bm {h}^{\tgt }_t, \bm {K} = \bm {W}_K \cdot \bm {h}^{\retr }_t, \bm {V} = \bm {W}_V \cdot \bm {h}^{\retr }_t. \end {split} \!\!
(4)
The outputs from the adapter are then added to the raw tar-
get H-space features as follows:
  \bm
 
{ h}_t
^ { \ retr } 
= \bm {
h }
^
{\
t
gt 
}_t + \lambda \underbrace {g_\varphi (\bm {h}^{\tgt }_t, \bm {h}^{\retr }_t)}_{\Delta \bm {h}_t}, 
(5)
where λ is a weighting parameter. We empirically find that
the results are not sensitive to the choice of λ, and set it
to be the cosine similarity between the retrieved and target
text CLIP embeddings. The intuition is that if the retrieved
content has a high correlation to the target prompt, it should
contribute more to the final generated content.
3.4. Training Procedure
Few-step diffusion models, trained through either self-
consistency [30, 53] or adversarial training [43, 44], op-
erate on a subset of n timesteps Tn = {t1, t2, · · · , tn}
drawn from the complete N timesteps in the teacher
model. For example, a 4-step student model uses Tn =
{1.0, 0.75, 0.5, 0.25}. The predicted ˆz0 at timestep ti, i ∈
{1, · · · , n} of the proposed denoiser reads
 \h a t {\bm {z} }_0 = f_\ theta
 (\bm {z}_{t_i}, t_i, \tau _\phi (\text {p}^{\tgt }), \bm {h}^{\retr }_0) .
In this work, we focus on latent adversarial train-
ing [43], as it empirically shows better performance than
self-consistency training, especially in the one-step sce-
nario. In addition, latent adversarial training, which op-
erates on the latent Z space, is more efficient than its
predecessor that performs adversarial training in the pixel
space [44].
Without loss of generality, we denote the teacher and
few-step student denoisers by fθ∗and fθ, respectively, both
frozen in our training schema. The trainable module for the
few-step denoiser is the H-space adapter. To further help
translate the adapted features into the final generation, we
also finetune the decoder of the student denoiser using the
parameter-efficient low-rank adaption tuning [15].
In our latent adversarial training, the discriminator (de-
noted by D) consists of the frozen UNet encoder of the
teacher denoiser (f enc
θ∗), followed by a few trainable pro-
jection layers that project multi-stage encoder features to
output embeddings (see Fig. 4). Unlike standard adversar-
ial training, latent adversarial training leverages the idea
of DiffusionGAN [56] that differentiates a pair of noisy
latent variables (ˆzs, zs), where ˆzs = αs ˆz0 + σsϵ and
zs = αsz0 + σsϵ, s ∼{t1, · · · , ts, · · · , tN}. It is worth
noting that the timesteps used to train the discriminator are
sampled uniformly from the full set of N timesteps such
that it can learn to distinguish samples across different noise
levels throughout the entire diffusion process.
Following [42–44], the hinge loss is used as the adver-
sarial objective function. The objective function for training
the discriminator is given as:
  
\be g
i
n
 {s
p
lit} \ ! \ ! \mat hcal {L}^ D_{\
t
e
xt {adv}} = \ sum _k \mat hbb { E}_{\bm {z}_0}\left [\max (0, 1 - D_k(\bm {z}_s, \tau _\phi (P^{\tgt }), t_s)) \right ] + \\ \mathbb {E}_{\hat {\bm {z}}_0} [ \max (0, 1 + D_k(\hat {\bm {z}}_s, \tau _\phi (P^{\tgt }), t_s)) ], \end {split}
(6)
where D stands for discriminator including the frozen
teacher encoder and trainable projection layers, and k de-
notes the k-th stage features. Similarly, the objective func-
tion for training the few-step denoiser, denoted by G fol-
lowing the convention of adversarial training, reads:
  
\ma t h
c
a
l {L}^G_{\t ext {adv}} = - \sum _k \mathbb {E}_{\bm {z}_0} [D_k(\hat {\bm {z}}_s, \tau _\phi (\text {p}^{\tgt })]. 
(7)
To stabilize the adversarial training procedure, we apply
spectral normalization [33] to the convolutional layers in the
trainable projection heads of the discriminator (see Fig. 4).
Compared to the gradient penalty method used in [18, 44],
spectral normalization does not require double backpropa-
gation, and hence it saves computations.
In addition to the adversarial training objective, we use
two auxiliary objectives to stabilize training and improve
image quality: i) score distillation loss, and ii) latent LPIPS
loss [18]. The score distillation loss is computed as the
smoothed L1 loss between ˆz0 and z0:
  \mathc a
l
 {L } _{\t e xt {
di
st ill} } = \ b
egin {ca s es} 
0.5* \| \hat {\bm {z}}_0 - \bm {z}_0 \|_2^2, \quad \text {if $|\hat {\bm {z}}_0 - \bm {z}_0| < 1$} \\ | \hat {\bm {z}}_0 - \bm {z}_0 | - 0.5, \quad \text {otherwise}. \end {cases} 
(8)
533

---

## Page 6

Similar to the LPIPS loss, the latent LPIPS loss is computed
in the latent space between ˆz0 and z0:
  \math c al
 
{L}_ {\text {el pips}}
 
=
 \mathbb {E}_{\mathcal {T}} \left [ \| F(\mathcal {T}(\hat {\bm {z}}_0)) - F(\mathcal {T}(\bm {z}_0) \|_2^2 \right ], 
(9)
where T (·) stands for a set of random differentiable aug-
mentations, including general geometric transformations
and cutoff, and F(·) denotes the embedding network (e.g.,
VGG16 [50]).
The final training objective is the weighted sum of the
aforementioned three objectives:
  \mat h cal {L} = \mathcal {L}_{\text {adv}} + \alpha \mathcal {L}_{\text {distill}} + \beta \mathcal {L}_{\text {latentLPIPS}}, 
(10)
where α and β are weight-balance parameters. However,
the extensive search of these parameters is expensive in
practice. Following [18, 44], we empirically set α = 2.5
and β = 1.0 in this paper.
4. Experiments
4.1. Experimental Setup
Datasets. We finetune the proposed method using a mix
of synthetic and real text-image pairs, as prior studies have
shown that synthetic data can help improve prompt fi-
delity [43].
We generate synthetic data via the teacher
model (i.e., Stable Diffusion v2-1-base [40]) at a constant
classifier-free guidance [12] value of 7.5, using around 3
million prompts from LAION-Aesthetic 6.25+ dataset [45],
which is commonly used for few-step diffusion distilla-
tion [57]. For the real data, we select 500K images from the
LAION-Aesthetic 5.5+ dataset [45] after filtering out those
with resolutions smaller than 1024 × 1024. Both the real
and synthetic images are subsequently resized to 512 × 512
and fed into the VAE encoder to obtain 64 × 64 latent em-
beddings.
Evaluation benchmarks.
We evaluate our proposed
method on two widely used benchmarks for evaluating text-
to-image generation: i) MS-COCO [2] and ii) TIFA. Specif-
ically, the MS-COCO benchmark comprises 5, 000 text-
image pairs from the validation set of the MS-COCO 2017
dataset, while the TIFA benchmark consists of 4, 081 text
prompts that contain elements from 12 categories (e.g., ob-
ject, spatial, activity, counting).
Evaluation metrics. Our quantitative evaluation employs a
set of metrics to assess different aspects of the generated im-
ages, including photorealism and faithfulness. For measur-
ing the photorealism of the generated images, we consider
the widely-adopted FID score [11] on the COCO bench-
mark, which measures if the generated images follow the
same distribution of the real images. In contrast, due to
the absence of reference images, we use Aesthetics (AES)
score to measure the visual appeal on the TIFA bench-
mark. To assess the faithfulness of the generated images
to the prompts, we adapt the widely used CLIP score [10]
on the COCO benchmark.
We compute CLIP score us-
ing the OpenCLIP-ViT-bigG-14 model. Although the
CLIP score provides an overall assessment of text-to-image
alignment, it may fail to provide an accurate evaluation of
specific object attributes, quantities, and spatial relation-
ships in the generated images. Therefore, we use the TIFA
score [17] to measure if the generated images can truthfully
reflect the target prompts by visual question answering.
Baselines. To ensure a fair comparison, we compare the
proposed method with several popular text-to-image LDM
baselines with similar model sizes. In particular, we con-
sider stable diffusion models [40], such as Stable Diffusion
v1-5 and Stable Diffusion v2-1-base. In addition, we com-
pare our model with few-step diffusion models, including
the Latent Consistency Model (LCM) [30] and Stable Dif-
fusion Turbo v2-1-base trained by latent adversarial distil-
lation [43] rather than pixel-space adversarial distillation as
in [44]. We also compare our method to RDM [1], which in-
tegrates retrieval augmentation into stable diffusion models.
To ensure a fair comparison, our method and all baselines
have almost the same model size. For the distilled few-step
models, the classifier-free guidance [12] is disabled, as it
will easily overshoot [43]. For the other baseline models,
we use a constant classifier-free guidance value of 7.5.
Implementation details.
To obtain retrieved text-image
pairs we apply the ScaNN search algorithm [8] in the text
feature space of a pretrained CLIP text encoder [37]. Here,
we reuse the same CLIP text features used by the stable
diffusion model (i.e., OpenCLIP-ViT-H-14) without in-
troducing any additional computational burden during in-
ference. Other retrieval methods, such as the lexical prompt
retriever (e.g., BM25 retriever) can be applied in different
applications without introducing expensive computational
overhead.
We use the 0.63M text-image pairs from the
OpenImage database as the retrieval database. Our finetun-
ing is performed on 64× NVIDIA L40S GPUs with a total
batch size of 2, 048. We finetune the 1-step stable diffusion
turbo model for 20K iterations using the AdamW optimizer
with a constant learning rate of 1 × 10−5 for both few-step
student model and the discriminator. The overall finetuning
process takes approximately one week.
4.2. Results on the MS-COCO Benchmark
Our evaluation on the MS-COCO benchmark demonstrates
ImageRAGTurbo’s superior performance (see Table 1).
Compared to the baseline Stable Diffusion Turbo v2-1-base
without retrieval augmentation, our approach achieves bet-
ter text-image alignment with a CLIP score that is 1.37%
higher, and improved image quality as indicated by a lower
FID score.
Our 1-step ImageRAGTurbo even surpasses
the 4-step latent consistency model by a large margin in
both CLIP and FID scores. Furthermore, our 1-step model
534

---

## Page 7

Table 1.
Quantitative comparison of text-to-image generation
models on the MS-COCO benchmark measured by the FID and
CLIP scores, as well as the number of function evaluations (NFE)
in the denoising process.
Models
NFE
FID↓
CLIP↑
w/o Retrieval
Stable Diffusion v1-5
50
24.38
0.319
Stable Diffusion v2-1
50
25.33
0.330
Latent Consistency Model
4
36.52
0.307
Stable Diffusion Turbo v2-1
1
26.04
0.319
w/ Retrieval
RDM
50
27.60
0.293
ImageRAGTurbo (Ours)
1
25.59
0.323
Table 2.
Quantitative comparison of text-to-image generation
models on the TIFA benchmark as measured by the TIFA and AES
scores, and the number of function evaluations (NFE) in denoising.
Models
NFE
AES↑
TIFA↑
w/o Retrieval
Stable Diffusion v1-5
50
5.79
0.768
Stable Diffusion v2-1
50
6.04
0.811
Latent Consistency Model
4
5.80
0.764
Stable Diffusion Turbo v2-1
1
5.85
0.779
w/ Retrieval
RDM
50
5.40
0.725
ImageRAGTurbo (Ours)
1
5.88
0.801
achieves comparable results to the 50-step teacher model
(i.e., Stable Diffusion v2-1-base), showing a drop of 1.03%
and 2.1% in FID and CLIP scores, respectively.
These
improvements can be attributed to ImageRAGTurbo’s en-
hanced ability to retrieve and leverage fine-grained, high-
quality reference images for more precise image synthesis.
In comparison to other retrieval-augmented approaches, Im-
ageRAGTurbo shows substantial improvement over RDM
with a CLIP score that is 10% higher, which highlights the
effectiveness of the proposed H-space adapter tuning.
4.3. Results on the TIFA Benchmark
As shown in Table 2, ImageRAGTurbo also demonstrates
superior performance on the TIFA benchmark. Compared
to the baseline Stable Diffusion Turbo v2-1-base without
retrieval augmentation, our approach achieves better text-
image fidelity with a TIFA score that is 2.2% higher, and a
slightly better aesthetic quality as indicated by the aesthetic
score (5.88 vs 5.83). Additionally, our 1-step ImageRAG-
Turbo outperforms the 4-step latent consistency model by a
large margin, showing a 3.7% improvement in TIFA score.
Although our 1-step model shows an overall drop of 1.2%
in TIFA score compared to 50-step Stable Diffusion v2-
1-base model, it achieves a comparable or even slightly
Figure 5. Detailed histogram of TIFA scores across various cate-
gories. Despite still lagging behind 50-step Stable Diffusion v2-1-
base model, our 1-step ImageRAGTurbo achieves comparable or
even slightly higher TIFA score in certain categories such as ob-
ject, activity, and material.
[50-step] 
SD
[1-step] 
ImageRAGTurbo
[1-step] 
SD Turbo
[4-step] 
LCM
TIFA Score = 0.811
Latency = 2960ms
TIFA Score = 0.801
Latency = 116.7ms
TIFA Score = 0.779
Latency = 113.8ms
TIFA Score = 0.764
Latency = 220.6ms
Inference Time (ms/image)
~25x faster &
~1.2% drop
~3% better
~2x faster & ~4.8% better
Figure 6. Comparison of inference time (ms/image). We report
the average inference time calculated over the same set of 100
prompts at 512 × 512 resolution on a single NVIDIA L40S GPU.
higher TIFA score in certain categories such as object, loca-
tion, activity, and material (see Fig. 5). Compared to other
retrieval-augmented approaches, ImageRAGTurbo substan-
tially outperforms RDM with a TIFA score 7.6% higher,
and a better aesthetic quality (5.88 vs 5.40), while requiring
only a single inference step instead of 50 steps.
4.4. Efficiency Analysis
Fig. 6 shows the computational efficiency of our method
compared to baselines.
The total inference time of our
method (116.7ms) is on par with that of the Stable Dif-
fusion (SD) Turbo (113.8ms). The inference time of our
method can be broken down into two components: retrieval
(1.9ms/image) and denoising (114.8ms/image). The 1 ms
increase in denoising latency compared to SD Turbo is at-
tributable to the H-space adapter, with 36M additional pa-
rameters, representing only 4% of the total model param-
eters. Our method introduces only a minimal overhead of
2.5% in latency compared to the SD Turbo, due to the added
535

---

## Page 8

(a) Representative samples generated by ImageRAGTurbo.
(b) Qualitative comparison with baseline methods.
Figure 7. Qualitative results of ImageRAGTurbo. (a) ImageRAGTurbo produces high-quality samples across diverse prompts and visual
concepts with a single denoising step. (b) ImageRAGTurbo achieves better text-to-image alignment than other competing few-step methods
on a variety of scenes, objects, and compositions.
retrieval mechanism and adapter, while delivering a signif-
icant 3% improvement in TIFA score.
Additionally, the
proposed method is approximately 25× faster than the SD
teacher model, with only a 1.2% drop in TIFA score. Com-
pared to the latent consistency model, our method demon-
strates a 3.7% higher TIFA score with around 47% more
latency (116.7ms vs 220.6ms).
4.5. Qualitative Analysis
To highlight the effectiveness of our proposed model and
its potential applications, we present qualitative examples in
Fig. 7. Specifically, our approach can generate high-quality
images in a single step (see Fig. 7a). Notably, our approach
outperforms other few-step baselines in text-to-image align-
ment (see Fig. 7b). In particular, our RAG-based proposed
method can generate accurate objects (demonstrated in ex-
amples/rows 2 and 3) and proper spatial relationship (shown
in examples/rows 1 and 4).
5. Conclusion and Future Work
In this paper, we have presented ImageRAGTurbo, a novel
retrieval-augmented framework designed to enhance few-
step diffusion models through an efficient H-space adapter
tuning. Our approach tackles a key challenge in few-step
diffusion models (insufficient text-to-image alignment) by
effectively leveraging relevant information from a large re-
trieval database. Through this efficient retrieval-augmented
mechanism, ImageRAGTurbo can improve the model’s
ability to generate images that faithfully reflect the input
prompts without compromising the inference latency. Ex-
perimental results show that our method significantly im-
proves text-to-image alignment without compromising im-
age quality.
While our framework opens up a new direction for
retrieval-augmented few-step diffusion models, several av-
enues remain for future investigation.
First, our current
framework relies on CLIP-based retrieval, while incorpo-
rating more fine-grained strategies, such as compositional
retrieval, is a promising direction for future work. Second,
we evaluate the current framework only on UNet-based ar-
chitectures, and extending the proposed method to diffu-
sion transformers (DiTs) [35] is worth exploring. Similar
to the H-space in UNets, recent studies suggest that DiTs
also learn semantically structured latent representations in
deeper layers, where structured and disentangled features
emerge in latent patch spaces and attention maps [9, 49].
536

---

## Page 9

References
[1] Andreas Blattmann, Robin Rombach, Kaan Oktay, Jonas
M¨uller, and Bj¨orn Ommer. Semi-parametric neural image
synthesis. arXiv preprint arXiv:2204.11824, 2022. 2, 3, 6
[2] Holger Caesar, Jasper Uijlings, and Vittorio Ferrari. Coco-
stuff: Thing and stuff classes in context, 2018. 6
[3] Wenhu Chen, Hexiang Hu, Chitwan Saharia, and William W
Cohen. Re-imagen: Retrieval-augmented text-to-image gen-
erator. arXiv preprint arXiv:2209.14491, 2022. 2, 3
[4] Quan Dao, Hao Phung, Trung Tuan Dao, Dimitris N
Metaxas, and Anh Tran.
Self-corrected flow distillation
for consistent one-step and few-step image generation. In
AAAI Conference on Artificial Intelligence, pages 2654–
2662, 2025. 2
[5] Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley,
Alex Chao, Apurva Mody, Steven Truitt, Dasha Metropoli-
tansky, Robert Osazuwa Ness, and Jonathan Larson. From
local to global: A graph rag approach to query-focused sum-
marization. arXiv preprint arXiv:2404.16130, 2024. 2
[6] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim
Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik
Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling recti-
fied flow transformers for high-resolution image synthesis.
In Int. Conf. Machine Learning, 2024. 3
[7] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing
Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and
Yoshua Bengio. Generative adversarial networks. Commu-
nications of the ACM, 63(11):139–144, 2020. 2
[8] Ruiqi Guo, Philip Sun, Erik Lindgren, Quan Geng, David
Simcha, Felix Chern, and Sanjiv Kumar. Accelerating large-
scale inference with anisotropic vector quantization, 2020.
6
[9] Alec Helbling, Tuna Han Salih Meral, Ben Hoover, Pinar
Yanardag, and Duen Horng Chau. Conceptattention: Diffu-
sion transformers learn highly interpretable features. arXiv
preprint arXiv:2502.04320, 2025. 8
[10] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras,
and Yejin Choi. Clipscore: A reference-free evaluation met-
ric for image captioning. arXiv preprint arXiv:2104.08718,
2021. 6
[11] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner,
Bernhard Nessler, and Sepp Hochreiter. Gans trained by a
two time-scale update rule converge to a local nash equilib-
rium. Advances in neural information processing systems,
30, 2017. 6
[12] Jonathan Ho and Tim Salimans.
Classifier-free diffusion
guidance, 2022. 6
[13] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising dif-
fusion probabilistic models. Advances in neural information
processing systems, 33:6840–6851, 2020. 2, 4
[14] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffu-
sion probabilistic models, 2020. 1
[15] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-
Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen.
Lora: Low-rank adaptation of large language models, 2021.
5
[16] Hexiang Hu, Kelvin CK Chan, Yu-Chuan Su, Wenhu Chen,
Yandong Li, Kihyuk Sohn, Yang Zhao, Xue Ben, Boqing
Gong, William Cohen, et al. Instruct-imagen: Image gen-
eration with multi-modal instruction.
In Proceedings of
the IEEE/CVF conference on computer vision and pattern
recognition, pages 4754–4763, 2024. 3
[17] Yushi Hu, Benlin Liu, Jungo Kasai, Yizhong Wang, Mari
Ostendorf, Ranjay Krishna, and Noah A Smith. Tifa: Accu-
rate and interpretable text-to-image faithfulness evaluation
with question answering. In Proceedings of the IEEE/CVF
International Conference on Computer Vision, pages 20406–
20417, 2023. 4, 6
[18] Minguk Kang, Richard Zhang, Connelly Barnes, Sylvain
Paris, Suha Kwak, Jaesik Park, Eli Shechtman, Jun-Yan Zhu,
and Taesung Park. Distilling Diffusion Models into Condi-
tional GANs. In European Conference on Computer Vision
(ECCV), 2024. 5, 6
[19] Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten,
Jaakko Lehtinen, and Timo Aila.
Analyzing and improv-
ing the image quality of stylegan.
In Proceedings of
the IEEE/CVF conference on computer vision and pattern
recognition, pages 8110–8119, 2020. 2
[20] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine.
Elucidating the design space of diffusion-based generative
models. Advances in neural information processing systems,
35:26565–26577, 2022. 4
[21] Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Ma-
tiana, Joe Penna, and Omer Levy.
Pick-a-pic: An open
dataset of user preferences for text-to-image generation,
2023. 2
[22] Mingi Kwon, Jaeseok Jeong, and Youngjung Uh. Diffusion
models already have a semantic latent space. arXiv preprint
arXiv:2210.10960, 2022. 4
[23] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni,
Vladimir Karpukhin, Naman Goyal, Heinrich
K¨uttler, Mike Lewis, Wen-tau Yih, Tim Rockt¨aschel, et al.
Retrieval-augmented generation for knowledge-intensive nlp
tasks. In Adv. Neural Inform. Process. Syst., pages 9459–
9474, 2020. 2
[24] Xiuyu Li, Yijiang Liu, Long Lian, Huanrui Yang, Zhen
Dong, Daniel Kang, Shanghang Zhang, and Kurt Keutzer.
Q-diffusion: Quantizing diffusion models. In IEEE Conf.
Comput. Vis. Pattern Recog., pages 17535–17545, 2023. 2
[25] Yanjing Li, Sheng Xu, Xianbin Cao, Xiao Sun, and
Baochang Zhang. Q-dm: An efficient low-bit quantized dif-
fusion model. In Adv. Neural Inform. Process. Syst., pages
76680–76691, 2023. 2
[26] Alexander Lobashev, Dmitry Guskov, Maria Larchenko, and
Mikhail Tamm. Hessian geometry of latent space in gener-
ative models. In Forty-second International Conference on
Machine Learning, 2025. 4
[27] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan
Li, and Jun Zhu. Dpm-solver: A fast ode solver for diffu-
sion probabilistic model sampling in around 10 steps. Ad-
vances in neural information processing systems, 35:5775–
5787, 2022. 1, 2, 4
[28] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan
Li, and Jun Zhu. Dpm-solver++: Fast solver for guided sam-
537

---

## Page 10

pling of diffusion probabilistic models. Machine Intelligence
Research, pages 1–22, 2025. 1, 2, 4
[29] Jinqi Luo, Tianjiao Ding, Kwan Ho Ryan Chan, Darshan
Thaker, Aditya Chattopadhyay, Chris Callison-Burch, and
Ren´e Vidal.
Pace: Parsimonious concept engineering for
large language models. Advances in Neural Information Pro-
cessing Systems, 37:99347–99381, 2024. 4
[30] Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang
Zhao.
Latent consistency models:
Synthesizing high-
resolution images with few-step inference. arXiv preprint
arXiv:2310.04378, 2023. 2, 4, 5, 6
[31] Yuanhuiyi Lyu, Xu Zheng, Lutao Jiang, Yibo Yan, Xin
Zou, Huiyu Zhou, Linfeng Zhang, and Xuming Hu.
Re-
alRAG: Retrieval-augmented realistic image generation via
self-reflective contrastive learning. In Forty-second Interna-
tional Conference on Machine Learning, 2025. 3
[32] Xinyin Ma, Gongfan Fang, and Xinchao Wang. Deepcache:
Accelerating diffusion models for free. In IEEE Conf. Com-
put. Vis. Pattern Recog., pages 15762–15772, 2024. 2
[33] Takeru Miyato, Toshiki Kataoka, Masanori Koyama, and
Yuichi Yoshida. Spectral normalization for generative ad-
versarial networks. arXiv preprint arXiv:1802.05957, 2018.
5
[34] Yong-Hyun Park, Mingi Kwon, Jaewoong Choi, Junghyo
Jo, and Youngjung Uh. Understanding the latent space of
diffusion models through the lens of riemannian geometry.
Advances in Neural Information Processing Systems, 36:
24129–24142, 2023. 4
[35] William Peebles and Saining Xie. Scalable diffusion models
with transformers. In Proceedings of the IEEE/CVF inter-
national conference on computer vision, pages 4195–4205,
2023. 8
[36] Dustin
Podell,
Zion
English,
Kyle
Lacey,
Andreas
Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and
Robin Rombach.
Sdxl: Improving latent diffusion mod-
els for high-resolution image synthesis.
arXiv preprint
arXiv:2307.01952, 2023. 3
[37] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya
Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,
Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning
transferable visual models from natural language supervi-
sion. In International conference on machine learning, pages
8748–8763. PmLR, 2021. 3, 6
[38] Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay, Am-
non Shashua, Kevin Leyton-Brown, and Yoav Shoham. In-
context retrieval-augmented language models. Transactions
of the Association for Computational Linguistics, 11:1316–
1331, 2023. 2
[39] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu,
and Mark Chen. Hierarchical text-conditional image gener-
ation with clip latents, 2022. 1
[40] Robin Rombach, Andreas Blattmann, Dominik Lorenz,
Patrick Esser, and Bj¨orn Ommer.
High-resolution image
synthesis with latent diffusion models.
In Proceedings of
the IEEE/CVF conference on computer vision and pattern
recognition, pages 10684–10695, 2022. 1, 3, 6
[41] Chitwan Saharia, William Chan, Saurabh Saxena, Lala
Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour,
Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans,
et al. Photorealistic text-to-image diffusion models with deep
language understanding.
Advances in neural information
processing systems, 35:36479–36494, 2022. 1, 3
[42] Axel Sauer, Kashyap Chitta, Jens M¨uller, and Andreas
Geiger. Projected gans converge faster, 2021. 5
[43] Axel Sauer, Frederic Boesel, Tim Dockhorn, Andreas
Blattmann, Patrick Esser, and Robin Rombach. Fast high-
resolution image synthesis with latent adversarial diffusion
distillation. In SIGGRAPH Asia 2024 Conference Papers,
pages 1–11, 2024. 2, 4, 5, 6
[44] Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin
Rombach. Adversarial diffusion distillation. In European
Conference on Computer Vision, pages 87–103. Springer,
2024. 2, 4, 5, 6
[45] Christoph Schuhmann, Romain Beaumont, Richard Vencu,
Cade Gordon,
Ross Wightman,
Mehdi Cherti,
Theo
Coombes, Aarush Katta, Clayton Mullis, Mitchell Worts-
man, Patrick Schramowski, Srivatsa Kundurthy, Katherine
Crowson, Ludwig Schmidt, Robert Kaczmarczyk, and Jenia
Jitsev. Laion-5b: An open large-scale dataset for training
next generation image-text models, 2022. 6
[46] Antonio Sclocchi, Alessandro Favero, and Matthieu Wyart.
A phase transition in diffusion models reveals the hierarchi-
cal nature of data. Proceedings of the National Academy of
Sciences, 122(1):e2408799121, 2025. 4
[47] Rotem Shalev-Arkushin, Rinon Gal, Amit H Bermano,
and Ohad Fried.
ImageRAG: Dynamic image retrieval
for reference-guided image generation.
arXiv preprint
arXiv:2502.09411, 2025. 2, 3
[48] Shelly Sheynin, Oron Ashual, Adam Polyak, Uriel Singer,
Oran Gafni, Eliya Nachmani, and Yaniv Taigman.
Knn-
diffusion: Image generation via large-scale retrieval, 2022.
2
[49] Zitao Shuai, Chenwei Wu, Zhengxu Tang, Bowen Song, and
Liyue Shen. Latent space disentanglement in diffusion trans-
formers enables precise zero-shot semantic editing. arXiv
preprint arXiv:2411.08196, 2024. 8
[50] Karen Simonyan and Andrew Zisserman. Very deep convo-
lutional networks for large-scale image recognition, 2015. 6
[51] Jiaming
Song,
Chenlin
Meng,
and
Stefano
Ermon.
Denoising diffusion implicit models.
arXiv preprint
arXiv:2010.02502, 2020. 1, 2, 4
[52] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Ab-
hishek Kumar, Stefano Ermon, and Ben Poole. Score-based
generative modeling through stochastic differential equa-
tions. arXiv preprint arXiv:2011.13456, 2020. 4
[53] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya
Sutskever. Consistency models. In International Conference
on Machine Learning, pages 32211–32252. PMLR, 2023. 2,
4, 5
[54] Pascal Vincent. A connection between score matching and
denoising autoencoders. Neural computation, 23(7):1661–
1674, 2011. 4
[55] Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou,
Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caim-
ing Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model
alignment using direct preference optimization, 2023. 2
538

---

## Page 11

[56] Zhendong Wang, Huangjie Zheng, Pengcheng He, Weizhu
Chen, and Mingyuan Zhou. Diffusion-gan: Training gans
with diffusion. arXiv preprint arXiv:2206.02262, 2022. 5
[57] Tianwei Yin, Micha¨el Gharbi, Taesung Park, Richard Zhang,
Eli Shechtman, Fredo Durand, and Bill Freeman.
Im-
proved distribution matching distillation for fast image syn-
thesis. Advances in neural information processing systems,
37:47455–47487, 2024. 6
[58] Huaying Yuan, Ziliang Zhao, Shuting Wang, Shitao Xiao,
Minheng Ni, Zheng Liu, and Zhicheng Dou. Finerag: Fine-
grained retrieval-augmented text-to-image generation.
In
Proceedings of the 31st International Conference on Com-
putational Linguistics, pages 11196–11205, 2025. 2, 3
[59] Wenliang Zhao, Lujia Bai, Yongming Rao, Jie Zhou, and
Jiwen Lu. Unipc: A unified predictor-corrector framework
for fast sampling of diffusion models. Advances in Neural
Information Processing Systems, 36:49842–49869, 2023. 1,
2, 4
[60] Zhenyu Zhou, Defang Chen, Can Wang, Chun Chen, and
Siwei Lyu.
Simple and fast distillation of diffusion mod-
els.
Advances in Neural Information Processing Systems,
37:40831–40860, 2024. 2
[61] Zhenyu Zhou, Defang Chen, Can Wang, Chun Chen, and
Siwei Lyu. Simple and fast distillation of diffusion models,
2024. 2
[62] Chang Zou, Xuyang Liu, Ting Liu, Siteng Huang, and Lin-
feng Zhang. Accelerating diffusion transformers with token-
wise feature caching. In The Thirteenth International Con-
ference on Learning Representations, 2025. 2
539