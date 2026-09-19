# Teaching a small AI to fact-check and to know when it doesn't know

*Project report, 2026-09-19. The case for continuing this project instead
of the math-reasoning version.*

## The short version

We are training a small AI model to check whether a claim is true. It
reads the claim, looks up Wikipedia, and answers with three things: a
verdict (true, false, or not enough information), the exact sentences its
answer rests on, and how sure it is.

The part that makes this a research project and not just a class exercise:
we use reinforcement learning (RL) to teach the model **how sure to be**,
**when to say "I don't know"**, and **that some mistakes are worse than
others**. Calling a lie "true" does more damage than doubting a true
claim, and the model should learn that.

This is not just a plan. The data, search engine and models are set up,
and we have run a full baseline on 900 claims. That baseline shows a
large, measurable problem that RL is suited to fix. Details below.

## What the system does, with a real example

Here is a claim from the FEVER dataset that we ran through the system:

> **Claim:** "Sidse Babett Knudsen is a conductor."

1. **Search.** The system searches 5.4 million Wikipedia pages and pulls
   the 5 most relevant ones. One is her page, which says:
   *"Sidse Babett Knudsen (born 22 November 1968) is a Danish actress who
   works in theatre, television, and film."*
2. **Judge.** A language model reads the claim and the retrieved
   sentences and answers in a fixed format: verdict, evidence sentences,
   confidence.
3. **Score.** Every answer is checked automatically against the dataset's
   answer key. No human grading and no AI judge.

What happened:

| | Verdict | Confidence it stated | Right? |
|---|---|---|---|
| Our small model (2 billion parameters, runs on our PC) | TRUE | 100% | ✗ |
| A large model (180 billion parameters, paid per use) | FALSE, citing the sentence above | 95% | ✓ |

The small model had the right evidence in front of it and still said
"true", with complete confidence. This is not a one-off. It is the
pattern across the whole test, and it is the problem the project exists
to fix.

## What we have already measured

We ran both models on 900 FEVER claims: 300 true, 300 false, 300 where
Wikipedia doesn't settle it. Full numbers are in
[`docs/stage0-results.md`](docs/stage0-results.md).

**1. The search works.** Plain keyword search found the right Wikipedia
page for 57% of claims. We added a simple step that matches names in the
claim to page titles, and that went up to **87.5%**. So in most cases the
model is shown the evidence it needs. The rest of the project can focus
on what the model does with it.

**2. There is a big, clear gap for training to close.** When the right
evidence was in front of them:

| | Correct |
|---|---|
| Large model | 95% |
| Our small model | 63% |

The small model *can* see the answer. It just doesn't use it well. A
32-point gap on a task with an automatic answer key is what RL is good
at closing.

**3. The small model calls almost everything true.** It says "true" to
94% of true claims, but also to many false ones: it catches only 30% of
false claims and 14% of the unclear ones. A fact-checker that lets false
claims through is worse than having no fact-checker, because it adds
false confidence.

**4. Neither model's stated confidence means anything.** The small model
says 100% on 97% of its true/false verdicts, and usually 0% when it
answers "not enough information", as if confidence meant "how much
evidence" rather than "how likely I am to be right". The large model
never goes below 90%, yet is right only 72% of the time.
Asking nicely in the prompt does not fix this. It has to be trained, and
training it is a core part of this project.

**5. It is cheap and fits our hardware.** The small model uses 3.8 GB of
our 16 GB graphics card. All the large-model runs so far cost **$0.36
in total**.

## Why not the math version

The earlier plan was the same idea applied to math problems: train a small
model with RL to solve math, reward it when the final answer is right. We
think that is the weaker project, for four reasons.

**1. Math is where this whole method started, and it has been done to
death.** GRPO, the RL method both versions use, was introduced in a math
paper (DeepSeekMath, arXiv 2402.03300) and made famous by DeepSeek-R1 on
math and code. Since then a large number of groups have published
"RL on math with a small model" papers and reproductions. A course
project doing the same thing has little room to say anything new.

**2. On math, it is hard to prove our training did anything.** A 2025
study ("Spurious Rewards", arXiv 2506.10947) trained Qwen math models
with *random* rewards, which carry no information about the right
answer, and still got **+21 points** on a standard math test, against
+29 with the real rewards. The same study found that Qwen *instruct*
models, which have already been through this kind of training, barely
improve further even with the correct rewards. Our small model is a Qwen
instruct model. On math we could see a small gain or none, and we could
not show it came from our reward design. A separate study (arXiv
2504.13837) found that RL on math mostly makes a model more consistent at
answers it could already reach, rather than teaching it anything new.

**3. Math has no "I don't know", and no mistake is worse than another.** Every
math problem has an answer, and a wrong answer is just wrong. So the most
interesting parts of our design have nothing to act on there:

| | Math | Fact-checking |
|---|---|---|
| Must find information first | No, everything is in the question | Yes, search a library of 5.4M pages |
| "Not enough information" is a valid answer | No | Yes, a third of the test set |
| Some mistakes cost more than others | No | Yes, calling a lie "true" is worse |
| Confidence matters to the user | Not much | A lot: people act on a fact-check |
| Must show its evidence | No | Yes, and we can check it sentence by sentence |

**4. It is easier to present.** Anyone can follow "is this claim true,
and why?" In a live demo the audience can suggest a claim and watch the
system search, cite and answer. A math demo needs the audience to follow
the math before they can tell whether the model did well.

## What would be new

We searched the recent research papers to find out what has already been
done:

- **Already done** (so we don't claim them as new): RL to make a
  fact-checker cite the right evidence (Veri-R1, arXiv 2510.01932);
  RL for multi-step fact-checking (ProFact, arXiv 2606.13262); RL that
  teaches models honest confidence and when to refuse, but only on
  question-answering, not fact-checking.
- **Not found anywhere:**
  - RL-trained confidence and "I don't know" for **fact-checking**.
  - A reward that treats **some mistakes as worse than others**.
  - A reward that charges for **each extra search**, so the model learns
    when looking further is worth it.

One caveat. This search was done with AI search assistants, and only
ProFact was read in full by us. Before we write "no one has done this"
in the final report, we will open the papers ourselves and confirm it.
[`docs/related-work.md`](docs/related-work.md) lists every paper.

## Why we are confident this will work out

The plan is built so that **every stage produces a result on its own**,
even if a later stage fails.

| Stage | What we do | What we get even if the next stage fails | Status |
|---|---|---|---|
| 0 | Run models with no training | A full baseline: accuracy, confidence, where errors come from | **Done** |
| 1 | Train the small model with RL to judge evidence and state honest confidence | Before/after comparison on accuracy, false claims caught, and confidence | Next |
| 2 | Train a small decision-maker that chooses when to search again, when to stop, and when to say "I don't know" | Accuracy vs. cost trade-off curve | After stage 1 |

The things that most often sink RL projects, we have already checked:

- **Will the model produce answers we can score?** Yes, 84–93% of the
  time before any training. RL needs this, or there is nothing to reward.
- **Is there room to improve?** Yes, 32 points, measured.
- **Is the data available with an answer key?** Yes, FEVER (185,000
  claims), plus SciFact and AVeriTeC to test on different kinds of
  claims. All three are downloaded.
- **Does it fit our hardware and budget?** Yes, 3.8 GB of GPU memory and
  under a dollar per full baseline run.

## Risks, stated plainly

- **RL might not close the gap on a 2B model.** Then we try the 4B
  version of the same model, which was the original pick. Stage 1 still
  gives an honest before/after result either way.
- **Some tools we need run best on Linux.** The fast training software
  (vLLM) and the model's fast code need Linux, so we will set up WSL
  (Linux inside Windows). This is a known setup task, not a research risk.
- **"Not enough information" is the hardest case for both models.**
  Better search made both models *more* likely to commit to an answer.
  This is also where our design should help most, and we will report it
  honestly either way.
- **One piece of the reward needs fixing before training.** As first
  written, the confidence reward pushes the model to always say 0% or
  100%. We have worked out the fix (a standard scoring rule called the
  Brier score) and will apply it before stage 1.

## Next steps

1. Fix the confidence reward.
2. Decide how to score evidence on "not enough information" claims,
   which have no correct evidence to compare against.
3. Set up WSL for training.
4. Stage 1: RL-train the small model, then rerun the same 900-claim
   test for the before/after comparison.

## Bottom line

The math version would repeat what many groups have already done, with
little chance of showing our method made the difference. The
fact-checking version has a measured problem (a small model that
confidently calls false claims true), a working setup that runs on our
own hardware for pennies, and research gaps we could not find anyone
filling. Every stage produces something we can show. We recommend we
continue with fact-checking.
