---
name: academic-writing
description: Check academic and technical prose for structural soundness, tense conventions by section, and calibrated hedging. Reviews whether each section does its job, whether verb tense follows discipline convention (past for method and results, present for established knowledge and interpretation), and whether claims are hedged to match their evidence, flagging both over-claiming and over-hedging. Use when the user asks to check structure, tense, hedging, voice, or academic register in a paper, thesis chapter, report, or grant application. Trigger on "check my structure", "is my tense consistent", "am I over-claiming", "does this hedge too much", "academic writing check", "review the prose in my paper", "is this the right register".
---

# Academic writing check

Three passes over a technical document: structure, tense, hedging. Report findings by location. Do not rewrite unless asked.

## Pass 1: Structure

Ask what each section is for, then whether it does that job and only that job.

| Section | Job | Common failure |
|---|---|---|
| Abstract | Problem, method, headline result with a number, significance | No number; method omitted; reads as a table of contents |
| Introduction | Why the problem matters, what is unknown, what this work does | Textbook background with no gap statement; contribution never stated |
| Related work | Position this work against prior work | Annotated bibliography with no positioning |
| Method | Enough to reproduce | Results leak in; justification missing for non-obvious choices |
| Experimental design | Data, splits, metrics, controls, repetitions | Buried in Method; splits or repetitions unstated |
| Results | What was observed | Interpretation smuggled in; numbers without uncertainty |
| Discussion | What it means, where it fails | Restates Results; limitations that cost nothing to admit |
| Limitations | Real threats to the conclusions | Generic gestures at sample size |
| Conclusion | What is now known that was not before | New material appears; or pure summary with no claim |

Check in addition:

- **The central claim is stated once, explicitly, early.** If a reader cannot point at the sentence, it is not there. Many otherwise strong reports never state theirs.
- **Load matches importance.** Measure how much space the headline result gets against how much a side observation gets. Inversions are common and cheap to fix.
- **Forward references resolve.** "As shown in Section 5" should point somewhere that shows it.
- **Every section earns its place.** A section that could be deleted without loss should be.
- **Ordering follows the specification** where one exists, or has a reason not to.

## Pass 2: Tense

Conventions differ by field, so infer the target from the document and stay consistent rather than imposing one house style. In the absence of a signal, these are the defaults for technical and scientific writing:

- **Established knowledge, definitions, standing facts:** present. *PCA maximises projected variance.*
- **Prior work, as a body:** present or present perfect. *Turk and Pentland show that. Several authors have argued.*
- **A specific prior study's actions:** past. *Belhumeur et al. compared Fisherfaces against Eigenfaces.*
- **What you did:** past. *Images were split by subject. We fitted the basis on training data.*
- **What you observed:** past. *Accuracy fell to 0.14 at the strongest ramp.*
- **What a figure or table does, now, on the page:** present. *Figure 3 shows the reconstructions. Table 1 reports accuracy against k.*
- **What your results mean:** present. *The leading components encode illumination rather than identity.*
- **Standing conclusions:** present. *A variance threshold is the wrong criterion for choosing k.*
- **Future or conditional work:** future or modal. *A Fisherface comparison would test this directly.*

The distinction that carries the most weight: a past-tense observation is a fact about one experiment, and a present-tense claim is a fact about the world. Moving between them is a substantive move, not a stylistic one. Flag every place where an author reports a single measurement in the present tense as though it were a general property, and every place where a genuine standing conclusion is buried in the past tense and reads as merely something that happened once.

Also flag mid-paragraph tense drift, and inconsistent voice where a document alternates between first person and passive within one section for no reason.

## Pass 3: Hedging

Hedging is calibration. Both directions are errors.

**Over-claiming.** Flag:
- Causal verbs on correlational evidence: *causes, drives, leads to, results in* where only an association was measured.
- *Proves, demonstrates conclusively, establishes* where one experiment on one dataset was run.
- *Significant, significantly* without a test. If it means "large", say large.
- Generalisation past the evidence: a claim about faces from one laboratory database stated as a claim about face recognition.
- Absolutes: *always, never, all, none, guarantees, optimal* where the support is empirical.
- A single measurement reported as a property. One timing run is not a speedup figure for the method.

**Over-hedging.** Flag:
- Stacked hedges: *may possibly suggest, could potentially indicate, it might be argued that perhaps.* One hedge per claim.
- Hedging a fact you measured directly. If the number is in your results file, state it.
- *It is thought that, some argue, it is generally accepted* with no attribution. Name who, or drop it.
- Apologetic framing that undercuts a result the evidence supports.

**Calibration check.** For each substantive claim, ask what evidence sits behind it, then whether the sentence's confidence matches. The strongest academic prose is confident exactly where the evidence is strong and explicitly uncertain where it is not, and it marks the boundary rather than applying one register throughout.

Give particular attention to claims a document has to make about its own limitations. A limitations section that admits only costless weaknesses is a hedging failure, not a structural one.

## Output

For each finding: location by line, the text as written, what convention or calibration it violates, and a suggested revision. Group by pass. Lead with the findings that change how a reader reads the argument, not with tense slips.

Note what is already well calibrated. Authors who are told only about problems tend to overcorrect.
