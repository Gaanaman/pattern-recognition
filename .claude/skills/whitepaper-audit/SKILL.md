---
name: whitepaper-audit
description: Audit a technical report, whitepaper, thesis chapter, or coursework submission against a written specification, for a named audience. Runs five independent lanes (spec coverage, claim provenance, method soundness, independent judge, presentation) and returns severity-ranked findings with locations and recommended fixes. Use when the user asks to audit, review, grade, assess, mark, or sanity-check a report or paper, especially against an assignment brief, rubric, RFP, or requirements document. Trigger on "audit this report", "check my paper against the brief", "would this pass review", "grade this submission", "review my whitepaper", "does this answer the question", "mark this against the rubric". Supports offline mode, where local files are the only evidence base and no web search is permitted.
---

# Whitepaper audit

Audit a document against a specification, from the standpoint of a specific reader.

## Before starting

Establish three things. Do not proceed on guesses about any of them.

**The specification.** The document that says what the work was supposed to do: assignment brief, rubric, RFP, requirements doc, reviewer guidelines. If the user has not supplied it, ask. Never reconstruct a specification from the report being audited, from a README written by the author, or from your own sense of what such a document usually asks for. A report checked against its own restatement of the requirements will always pass, which makes the audit worthless.

**The audience.** "A DSCD612 course examiner" and "a venture capital associate" want opposite things from the same paragraph. The audience sets the severity of every finding. Ask if not given.

**The mode.** Offline means local files are the only evidence. No web search, no fetching, no reliance on recalled facts about external sources. In offline mode, anything you cannot check against a file present on disk is reported as UNVERIFIED rather than assumed correct or assumed wrong.

Convert the document to markdown or plain text first if it is LaTeX, docx, or PDF. Audit the source when a source exists, since line numbers in the source are what the author will act on.

## The five lanes

Run them in this order. Each lane is independent, and a later lane never softens an earlier lane's finding.

### Lane 1: Specification coverage

Build an explicit map. For every numbered or lettered requirement in the specification, record:

- the requirement, quoted, not paraphrased
- where the document satisfies it, by section and line
- a verdict: MET, PARTIAL, MISSING, or MET ELSEWHERE

MET ELSEWHERE matters more than it looks. Coursework and deliverables often span a report plus code plus appendices. A requirement satisfied in a notebook but absent from the report is not a pass if the specification asked for it in the report. Say exactly where it lives and exactly where the specification expected it.

Read requirement verbs literally. "Display the mean face" is not satisfied by defining the mean face in an equation. "Compare visually and quantitatively" is two obligations, not one. "At least three" is a floor to be checked, not a suggestion. Verb-level misses are the most common real finding and the easiest for an author to fix once told.

Check the specification for general requirements that sit outside the numbered tasks. Preambles routinely carry obligations of their own, and authors who work through the lettered tasks often skip them.

### Lane 2: Claim provenance

Extract every quantitative claim in the document: accuracies, counts, timings, percentages, dimensions, error bars, speedups. For each, find the source of truth on disk, which is usually a results file, a saved log, or the code that produced it.

Report each claim as SUPPORTED (matches a source), MISMATCHED (contradicts a source, quote both), or UNVERIFIED (no source on disk). Never round a mismatch away. A number that appears in the report as 0.937 and in the results file as 0.9367 is fine; one that appears as 0.94 and 0.89 is a finding.

Watch for claims that are numerically right but rhetorically overreaching: a speedup measured once reported as a general property, a single-split result stated as though it were averaged, a percentage whose denominator shifts between mentions.

### Lane 3: Method soundness

For empirical work, check the things that invalidate results rather than merely weaken them:

- **Leakage.** Was anything fit on data it is later evaluated on? Check preprocessing, scalers, feature selection, and dimensionality reduction separately from the classifier. Fitting a transform on the full set before splitting is the classic case and often hides in a line that looks like setup.
- **Selection on test.** Were hyperparameters, thresholds, component counts, or stopping points chosen using the same data that produced the headline number? This is distinct from leakage and is missed far more often, particularly in reports that argue carefully about leakage elsewhere.
- **Baselines.** Is the comparison like for like? A baseline trained on different data, or tuned less carefully, inflates the contribution.
- **Variance.** Is a difference claimed that is smaller than the reported noise? Are error bars present at all, and do they come from repeated splits rather than a single run?
- **Reproducibility.** Fixed seeds, pinned versions, a stated path from code to every reported number.

State clean findings as plainly as dirty ones. "The basis is fit on training data only, at these lines" is a result the author needs.

### Lane 4: Independent judge

Give this lane a fresh subagent with no access to the other lanes' output. Pass it the specification's questions verbatim and the document. Ask it to assess as the named audience would, cold.

The point is to catch what familiarity hides. By the time lanes 1 to 3 finish, you know the document too well to notice that a central argument is never actually stated, or that a section answers a question nobody asked. A cold reader notices.

Prompt the judge for: whether each specification question is answered and how convincingly, the strongest and weakest parts, anything an examiner or reviewer would circle, and a verdict in the audience's own terms. Do not tell it what the other lanes found.

Treat the judge's output as one input, not a verdict. Where it contradicts a lane that checked files directly, the file-checking lane wins, and say so.

### Lane 5: Presentation

Mechanical, quick, and frequently productive:

- Every figure and table referenced in the text exists, and every one that exists is referenced. Orphan figures usually mean a requirement was answered in code and lost on the way into the report.
- Cross-references resolve. Captions describe the content rather than restating the title.
- Section ordering matches the specification's ordering where the specification implies one, or has a stated reason not to.
- Length against any stated limit.
- Notation is defined before use and stays consistent.

## Output

Rank findings by severity, judged against the audience:

**Blocking.** A specification requirement is unmet, a claim contradicts its source, or a method flaw invalidates a headline result.

**Material.** A requirement is met weakly, evidence is thinner than the claim, or a reader in the target audience would reasonably object.

**Minor.** Presentation, phrasing, consistency.

For each finding give the location by line, what the specification or evidence says, what the document says, and a concrete fix. Skip findings you cannot substantiate. A short audit that is entirely right beats a long one padded with speculation.

Close with what the document does well, specifically and briefly. An author needs to know which parts to leave alone.

## Default posture

Recommend, do not edit, unless the user has asked for changes. Report coverage honestly: if a lane could not run, say which and why rather than quietly dropping it.
