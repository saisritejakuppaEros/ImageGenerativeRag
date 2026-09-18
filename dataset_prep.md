# RAG Dataset Preparation — Summary from Your Papers

The papers in `papers/` take **four different approaches** to RAG data — from "just build a retrieval index" to "synthesize an entire training pipeline." Here is how each one actually did it.

---

## The Big Picture: Two Kinds of "RAG Datasets"

People usually mean one of two things:

1. **Retrieval corpus** — the image/text database you search at inference time (LAION, OpenImages, web cache, etc.)
2. **Training/eval dataset** — prompts + retrieved evidence + (sometimes) ground-truth outputs, used to train or benchmark the RAG system

Your papers split cleanly across both.

---

## 1. ImageRAG — Minimal Prep: Just a Retrieval Index

**Paper:** `5918_ImageRAG_Dynamic_Image_Re.pdf`

ImageRAG is **inference-only**. It does **not** train a RAG model. Dataset prep is basically:

| Step | What they did |
|------|----------------|
| **Retrieval corpus** | Random **350K subset of LAION** |
| **Indexing** | Pre-compute **CLIP ViT-B/32** embeddings for all images (once) |
| **At runtime** | VLM finds missing concepts → writes retrieval captions → CLIP retrieves top matches |

No custom training pairs. The "dataset" is just an external image bank + embeddings.

They also tested **specialized corpora** (ImageNet, Dogs, Flowers, Cars) and found domain-specific retrieval works better than generic LAION for fine-grained concepts.

---

## 2. ImageRAGTurbo — Finetune the Generator *With* Retrieval

**Paper:** `Qiu_ImageRAGTurbo_Towards_One-step_Text-to-Image_Generation_with_Retrieval-Augmented_Diffusion_Models_CVPR_2026_paper.pdf`

Here they need **two** datasets:

### A. Finetuning data (teach the model to use retrieval)

- **~3M synthetic** text–image pairs: SD v2.1 generates images from LAION-Aesthetic 6.25+ prompts
- **500K real** images from LAION-Aesthetic 5.5+ (filtered to ≥1024×1024)
- All resized to 512×512, encoded to latents

### B. Retrieval database (what to retrieve at train/inference time)

- **630K text–image pairs from OpenImages**
- Retrieved via **ScaNN** on CLIP text features (same encoder as SD)
- Retrieval branch features are **pre-cached** for efficiency

So: standard LAION for training the diffusion model, OpenImages as the retrieval bank.

---

## 3. Gen-Searcher — Full Synthetic Pipeline (Most Elaborate)

**Paper:** `2603.28767v3.pdf`

Gen-Searcher needs **aligned triples**: search-heavy prompt → agent trajectory → grounded image. That data doesn't exist naturally, so they built a **4-stage pipeline**:

```
1. Prompt construction
   └─ Gemini 3 Pro generates multi-hop search prompts (~20 domains)
   └─ OR convert DeepResearch QA datasets into image-gen prompts

2. Agentic trajectory generation
   └─ Gemini 3 Pro + tools (text search, image search, browse)
   └─ Multi-turn: search → analyze → refine → grounded prompt + reference images

3. Ground-truth image synthesis
   └─ Nano Banana Pro generates target images from grounded prompts
   └─ ~30K raw samples

4. Filtering & curation
   └─ Seed1.8 scores: faithfulness, aesthetics, safety, search necessity
   └─ Rule-based filters (length, consistency)
   └─ ~17K high-quality → split into:
       • Gen-Searcher-SFT-10k
       • Gen-Searcher-RL-6k
       • KnowGen benchmark (630 human-verified, held out)
```

**Key idea:** use a strong proprietary stack (Gemini + Nano Banana) to *manufacture* supervised data, then filter aggressively.

---

## 4. SearchGen — Start From Real User Failures

**Paper:** `2607.05382v4.pdf`

SearchGen starts from **production traffic**, not synthetic prompts:

### Layer 1: Seed entities

- **20,840 real user prompts** from production AIGC
- Annotators label failure modes → **12 categories** (temporal, entity/IP, cultural, etc.)
- Extract **31,537 entities** with canonical names, frequency estimates, visual refs, attributes
- **93.1% of entities appear only once** → long-tail by design

### Layer 2: Prompt synthesis

- **Template instantiation** from seed entities (parameterized slots)
- **LLM rewrite** into natural user-style prompts (preserve entities + modality needs)
- **QC:** remove ambiguous/insensitive prompts; verify checklist items are answerable
- Each prompt gets: 2–4 search queries, 3–8 visual elements, binary checklists, 1–5 rubric

### Layer 3: Frozen search corpus (the clever part)

- Pre-run **159K search sessions** → **SEARCHGEN-CORPUS-1M**
- **705K cached downloads**, **1.1M URLs**
- Lets you replay search **offline** without live APIs — critical for RL/preference learning

Final scale:

- **20,188 training prompts** (~5.2 knowledge gaps each)
- **96,848 reasoning traces**
- **283,493 generated images**
- **751 test prompts** (SEARCHGEN-BENCH)

---

## Side-by-Side Comparison

| Paper | Retrieval corpus | Training/eval data | Main trick |
|-------|------------------|-------------------|------------|
| **ImageRAG** | LAION 350K (+ CLIP index) | None (inference only) | VLM diagnoses gaps at runtime |
| **ImageRAGTurbo** | OpenImages 630K | LAION synthetic + real for finetuning | Pre-cache retrieval features |
| **Gen-Searcher** | Live web (Gemini tools) | Synthetic pipeline → SFT-10k / RL-6k | Strong models generate + filter data |
| **SearchGen** | Pre-cached web (1M corpus) | Production failures → templates → LLM rewrite | Answer-first + frozen search replay |

---

## Common Patterns Across All Four

1. **CLIP (or similar) for retrieval** — text/image similarity is the default retriever
2. **Long-tail is the point** — rare entities, current events, fine-grained concepts
3. **Two modalities** — visual refs *and* textual knowledge (flags, dates, UI specs)
4. **Corpus quality > size** — specialized datasets beat generic LAION for niche concepts
5. **Pre-compute when possible** — embeddings, search results, retrieval branch features
6. **Synthetic data when real pairs don't exist** — Gen-Searcher and SearchGen both manufacture training data from LLMs + filters

---

## If You're Building Your Own Image RAG Dataset

A practical recipe combining these ideas:

```
Phase 1 — Retrieval corpus
  • Pick base: LAION subset, OpenImages, or domain-specific images
  • Pre-compute CLIP/SigLIP embeddings
  • Optional: metadata (captions, tags, source)

Phase 2 — Prompts that need RAG
  • Mine failure cases (like SearchGen), OR
  • LLM-generate search-intensive prompts (like Gen-Searcher), OR
  • Use fine-grained benchmarks (ImageNet tail, iNaturalist, etc.)

Phase 3 — Evidence pairing
  • Runtime: VLM finds gaps → retrieve (ImageRAG)
  • Offline: run search once, cache results (SearchGen)
  • Full pipeline: agent trajectories + GT images (Gen-Searcher)

Phase 4 — Filter
  • Model-based scoring (faithfulness, aesthetics)
  • Human verification on benchmark holdout
  • No train/test overlap on entities
```

The main fork is **training vs. inference-only**: ImageRAG needs almost no dataset prep; Gen-Searcher/SearchGen invest heavily in curated training data because they're training search agents, not just retrieving at inference.
