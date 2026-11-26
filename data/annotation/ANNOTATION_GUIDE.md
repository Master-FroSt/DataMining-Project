# Big Walk Steam Reviews: 3-Class Stance Annotation Guide

This guide defines the manual coding rules for labeling `gold_label` in `data/annotation/annotation_candidates.csv` (saving verified labels to `data/annotation/gold_labels.csv`).

## Target Classes

Assign exactly one of the three mutually exclusive labels to each review based on its primary communicative intent:

1. `Gameplay_Genuine`
   - **Definition**: Reviews evaluating the game itself (co-op mechanics, proximity voice chat, puzzles, world exploration, audio/visual style, bugs, performance, pricing, or playtime experience) without centering on the November 2026 GenAI/IGF controversy.
   - **Include**:
     - Short gameplay praise or complaints (e.g., `"Great co-op game with friends"`, `"Fun puzzles but optimization is rough"`).
     - Playtime-grounded bug reports or feature critiques unrelated to AI tooling.
   - **Exclude**:
     - Reviews whose main reason for recommending or not recommending is either criticizing the developers' AI disclosure or defending the developers against review bombing.

2. `Anti_AI_Backlash`
   - **Definition**: Reviews criticizing, rejecting, or warning against the game due to generative AI / LLM usage, the IGF submission eligibility controversy, perceived dishonesty/lack of disclosure, or "AI slop" concerns.
   - **Include**:
     - Explicit backlash (`"AI slop"`, `"0.1% is still GenAI"`, `"Used Claude/LLM code"`, `"Should be disqualified from IGF"`, `"Refunded after reading the FAQ"`).
     - Short negative boycott statements referencing AI or the controversy (`"No AI slop"`, `"AI garbage, refunded"`).
     - Negative reviews that bundle brief gameplay criticism with AI backlash as the decisive factor (`voted_up=False`).

3. `Pro_Dev_Clarification`
   - **Definition**: Reviews explicitly engaging with the November 2026 controversy in order to defend House House, contextualize the technical scope (`10 lines` of code, `0.1%`, `shadow caster` shader debugging), or push back against review bombing / misinformation.
   - **Include**:
     - Explicit counter-narrative defenses (`"Countering the review bomb"`, `"It was literally 10 lines of a shadow caster shader"`, `"People don't read the FAQ"`).
     - Reviews that use strong or angry vocabulary directed at *review bombers* while voting `voted_up=True` (key VADER-vs-rating divergence cases).

---

## Tie-Breaking & Edge-Case Rules

| Edge Case | Coding Rule | Assigned Class |
| :--- | :--- | :--- |
| Angry/profane tone praising the game while attacking review bombers (`voted_up=True`, e.g., `"Idiots review bombing a great indie game over 10 lines of shader code"`) | Intent is defending the devs / countering the backlash, despite negative VADER lexicon words (`idiots`, `bombing`). | `Pro_Dev_Clarification` |
| Sarcastic recommendation (`voted_up=True`) mocking the game or AI usage (`"10/10 best 10 lines of AI slop ever made, refunded"`) | Primary stance opposes the AI usage despite `voted_up=True` click. | `Anti_AI_Backlash` |
| Mentions AI briefly in passing (`"Heard there was drama, ignored it, really fun co-op puzzles with my friends"`) | Primary evaluation is genuine gameplay experience; controversy mention is incidental. | `Gameplay_Genuine` |
| Neutral/confused question about the controversy with `voted_up=False` (`"Not buying until they clarify the IGF and AI situation"`) | Withholding recommendation specifically due to the AI/IGF controversy. | `Anti_AI_Backlash` |
| Mixed gameplay critique + AI critique (`"Walking is slow and using gen AI makes me not trust the studio"`) | When AI/trust is an explicit reason for `voted_up=False`, code as backlash. | `Anti_AI_Backlash` |
| Pre-controversy (`P0_Pre_Controversy`) praise/critique with no AI reference | Cannot be part of the Nov 9+ controversy unless edited post-trigger (`was_edited=True` and mentions AI). | `Gameplay_Genuine` |

---

## Workflow for Annotators

1. Read `text_vader` (preserves capitalization, punctuation, and tone) alongside `voted_up`, `phase`, `playtime_at_review_hrs`, and `was_edited`.
2. Treat `suggested_label` ONLY as a weak pre-screening aid; override it whenever primary intent differs.
3. Fill `gold_label` with one exact string: `Gameplay_Genuine`, `Anti_AI_Backlash`, or `Pro_Dev_Clarification`.
4. Save the completed file as `data/annotation/gold_labels.csv` and switch `ALLOW_PRELABEL_DRYRUN = False` in `notebooks/big_walk_kdd_pipeline.ipynb`.
