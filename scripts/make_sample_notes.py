"""Generate 15 fake markdown notes mirroring the themes in idea.txt.

Used for smoke-testing the pipeline. Themes deliberately overlap with the
recurring topics Stefan named: AI agency, scams, work transition, body/training,
memory/physical media, games, plus a cross-cutting freedom-vs-structure tension.
One cluster has a deliberate spike (4 notes in 2 weeks in April 2026).
Another is stale (last seen late 2024).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

NOTES: list[dict] = [
    # --- AI agency / agent workflow (cluster A; spike in April 2026) ---
    {
        "filename": "2024-08-12-agent-design.md",
        "date": "2024-08-12",
        "title": "Agent design — what I keep coming back to",
        "tags": ["ai", "agents", "workflow"],
        "body": (
            "The thing about agents is that the design question is really a question "
            "about delegation. What can a model decide on its own, and what does it have "
            "to ask? I keep drawing the same picture: a loop of tool call, observation, "
            "decision. Most agent frameworks just hide the loop.\n\n"
            "The interesting part is what to do when the model is uncertain. Either you "
            "let it ask (interactive, slow) or you let it guess (autonomous, risky). "
            "Real work is somewhere in the middle, and the boundary shifts as the model "
            "gets more reliable."
        ),
    },
    {
        "filename": "2024-11-03-tooling-brain-dump.md",
        "date": "2024-11-03",
        "title": "Workflow tooling — brain dump",
        "tags": ["ai", "tooling", "automation"],
        "body": (
            "Three things I keep wanting: a single inbox for every async agent ping, a "
            "way to pause and resume a multi-step workflow without losing context, and "
            "an audit trail I can grep six months later.\n\n"
            "None of these exist cleanly in any of the current agent platforms. They all "
            "assume the agent runs to completion or it dies. Real consulting work is "
            "neither of those — it's ten starts, six pauses, and one finish three weeks "
            "later when the client finally approves the spec."
        ),
    },
    {
        "filename": "2026-04-02-ai-agency-pricing.md",
        "date": "2026-04-02",
        "title": "AI agency — how to actually price it",
        "tags": ["ai", "consulting", "pricing"],
        "body": (
            "Most AI consulting proposals I've seen price by the hour, which is exactly "
            "wrong. The whole point of the work is that the hour becomes meaningless when "
            "an agent does 80% of it. Better model: price the outcome, scope the journey.\n\n"
            "The hard part is that clients don't know what outcome they want yet. They "
            "have a problem and a budget, and the rest is discovery. So the price is for "
            "discovery plus delivery, with a clear off-ramp."
        ),
    },
    {
        "filename": "2026-04-09-agent-loop.md",
        "date": "2026-04-09",
        "title": "The agent loop, redrawn",
        "tags": ["ai", "agents"],
        "body": (
            "I keep redrawing this. The agent loop is not a loop. It's a tree. Each "
            "branch is a hypothesis, and the agent prunes branches that didn't pan out.\n\n"
            "If you design for the tree, the UI gets easier. You show the user the "
            "branches they didn't take. They learn from the pruning. That's the actual "
            "product — not the answer, the audit trail of how the answer was reached."
        ),
    },
    {
        "filename": "2026-04-15-workflow-bottleneck.md",
        "date": "2026-04-15",
        "title": "Workflow bottleneck is always me",
        "tags": ["ai", "workflow", "process"],
        "body": (
            "Every workflow I design eventually bottlenecks at the human approval step. "
            "Not because the human is slow — because the model can't tell what the human "
            "would approve without asking.\n\n"
            "Two ways out: richer context (so the model can predict approval), or richer "
            "defaults (so the human only vetoes, never initiates). Both are real work."
        ),
    },
    {
        "filename": "2026-04-21-agent-rollout.md",
        "date": "2026-04-21",
        "title": "Agent rollout — lessons from last week",
        "tags": ["ai", "rollout", "consulting"],
        "body": (
            "Rolled out the agent workflow to a client team this week. Three takeaways: "
            "(1) the demo took longer than the build, (2) the questions were about trust, "
            "not capability, (3) the team that uses it will not be the team that built it.\n\n"
            "I'm going to start billing for trust work separately from build work. "
            "Trust is the real product."
        ),
    },
    # --- Scams / online grifts (cluster B; STALE: last seen 2024-11) ---
    {
        "filename": "2024-08-21-crypto-grift.md",
        "date": "2024-08-21",
        "title": "Another crypto grift, noted",
        "tags": ["scams", "crypto", "online"],
        "body": (
            "Logged another one. The pattern is always the same: anonymous team, "
            "vague whitepaper, big promises, fake partnerships, countdown timer, "
            "exit liquidity.\n\n"
            "What gets me is that the people running these are not stupid. They "
            "know exactly what they're doing. The scariest part is that some of "
            "them have done this four or five times and each time the next batch "
            "of marks is bigger."
        ),
    },
    {
        "filename": "2024-09-15-newsletter-scam.md",
        "date": "2024-09-15",
        "title": "The newsletter grift",
        "tags": ["scams", "media", "online"],
        "body": (
            "There's a whole economy of people selling courses on how to sell "
            "courses. The newsletter grift is upstream of the course grift: build "
            "an audience, monetize the audience, then sell them the playbook you "
            "used to build it.\n\n"
            "Half the Substack creator economy is this. The other half is "
            "well-meaning people who are also making a living, which is fine, but "
            "the line is thin and the incentives push everyone toward the same "
            "kind of content."
        ),
    },
    {
        "filename": "2024-11-04-grifter-resume.md",
        "date": "2024-11-04",
        "title": "The grifter resume",
        "tags": ["scams", "career"],
        "body": (
            "I keep noticing the same resume pattern: someone has launched three "
            "or four 'AI startups' in eighteen months, none of which have users, "
            "all of which have a Twitter presence, and they're now 'raising a "
            "pre-seed round' for the next one.\n\n"
            "The tell is the LinkedIn banner — always a gradient, always a vague "
            "noun, always a tagline that could mean anything."
        ),
    },
    # --- Work transition / career uncertainty (cluster C) ---
    {
        "filename": "2025-01-08-freedom-vs-deadlines.md",
        "date": "2025-01-08",
        "title": "I want total freedom to follow curiosity",
        "tags": ["career", "freedom"],
        "body": (
            "I want total freedom to follow curiosity. That's the dream. The "
            "morning is mine, the project is mine, the next idea is mine.\n\n"
            "What I'm not sure about: whether the freedom is the point, or whether "
            "the freedom is the excuse I use to avoid the hard part, which is "
            "shipping something long enough to find out if it's good."
        ),
    },
    {
        "filename": "2025-03-14-needing-deadlines.md",
        "date": "2025-03-14",
        "title": "I need hard external deadlines or nothing ships",
        "tags": ["career", "process", "discipline"],
        "body": (
            "I need hard external deadlines or nothing ships. That's the other "
            "side of the same coin. The freedom is real, but so is the silence, "
            "and in the silence I write three drafts of a Substack post and publish "
            "none of them.\n\n"
            "External deadlines force a finish line. The finish line is the actual "
            "product. Without it, the work is just rehearsal."
        ),
    },
    {
        "filename": "2025-06-22-career-pivot.md",
        "date": "2025-06-22",
        "title": "Career pivot, third one this year",
        "tags": ["career", "transition"],
        "body": (
            "Third career pivot of the year. The previous two lasted six weeks. "
            "This one might last longer because the work is harder and the pay is "
            "worse, which is usually a sign I'm doing the right thing.\n\n"
            "Or it's a sign I'm running away again. Hard to tell from the inside."
        ),
    },
    # --- Body / training (cluster D) ---
    {
        "filename": "2025-02-18-training-log-feb.md",
        "date": "2025-02-18",
        "title": "Training log — February",
        "tags": ["body", "training"],
        "body": (
            "Hit a new squat PR this week. The setup is the same as three months "
            "ago: warm-up, work sets, cooldown. What changed is the patience to "
            "stay in the groove when the weight feels heavy.\n\n"
            "Patience is the actual variable. Not the program, not the diet. The "
            "ability to do the boring thing for a long time."
        ),
    },
    {
        "filename": "2025-08-30-sleep-and-recovery.md",
        "date": "2025-08-30",
        "title": "Sleep is the whole game",
        "tags": ["body", "recovery", "sleep"],
        "body": (
            "I keep relearning this. Two weeks of good sleep and everything else "
            "lines up: the training, the writing, the mood. Two weeks of bad sleep "
            "and I am a different person and not in a good way.\n\n"
            "The whole self-optimization industry exists because this is true and "
            "people want to skip the part where you go to bed at the same time "
            "every night."
        ),
    },
    # --- Memory / physical media (cluster E) ---
    {
        "filename": "2025-05-09-physical-notes-archive.md",
        "date": "2025-05-09",
        "title": "Going back to physical notes",
        "tags": ["memory", "physical", "notes"],
        "body": (
            "I archived all my old notebooks last weekend. The earliest is from "
            "2014. The handwriting changes over time but the obsession is the same: "
            "draw the thing, write the thing, leave the thing somewhere I can find "
            "it later.\n\n"
            "The notes app is faster and searchable but I don't remember the things "
            "I write in it. The notebooks I remember because I had to find a "
            "physical place to put them."
        ),
    },
    {
        "filename": "2026-01-17-cds-and-tapes.md",
        "date": "2026-01-17",
        "title": "CDs, tapes, and the weight of a thing",
        "tags": ["memory", "physical", "media"],
        "body": (
            "Bought a CD player this week. First time I've owned one since 2008. "
            "The thing I forgot about physical media is that it has weight, and "
            "weight changes how you relate to it.\n\n"
            "A playlist is a current state. An album is a decision someone made "
            "once and you experience it again. The decision is the part that "
            "matters."
        ),
    },
    # --- Games / Signal Lost (cluster F) ---
    {
        "filename": "2025-04-11-signal-lost-design.md",
        "date": "2025-04-11",
        "title": "Signal Lost — the design of a quiet game",
        "tags": ["games", "signal-lost", "design"],
        "body": (
            "I've been working on Signal Lost for about six months. The whole game "
            "is a single mechanic: you walk, you listen, you record. There's no "
            "score, no failure state, no dialogue.\n\n"
            "What I want the player to feel is the weight of paying attention. "
            "Most games make attention cheap — they reward it constantly. I want "
            "to make attention something you have to choose, and feel the cost of "
            "the choice."
        ),
    },
    {
        "filename": "2025-10-05-game-design-notes.md",
        "date": "2025-10-05",
        "title": "Why I keep coming back to quiet games",
        "tags": ["games", "design"],
        "body": (
            "Played through Outer Wilds again this month. Forgot how much the "
            "game trusts the player. There is no hand-holding, no quest log, no "
            "waypoints. The game is the player's attention, organized.\n\n"
            "That's the design lineage I want Signal Lost to be in. Quiet games "
            "that respect attention."
        ),
    },
    # --- An extra note, cross-cutting ---
    {
        "filename": "2026-05-30-finishing-things.md",
        "date": "2026-05-30",
        "title": "Maybe the problem is finishing things",
        "tags": ["career", "process"],
        "body": (
            "Maybe the problem is finishing things. I start a lot. I finish less. "
            "The list of finished work is short and the list of started work is "
            "long.\n\n"
            "The honest version is that I confuse starting with momentum. They "
            "are not the same. Starting feels like progress. Finishing is progress."
        ),
    },
]


def write_note(out_dir: Path, *, date: str, title: str, tags: list[str], body: str) -> Path:
    tags_line = "[" + ", ".join(tags) + "]"
    text = (
        "---\n"
        f"date: {date}\n"
        f"title: {title}\n"
        f"tags: {tags_line}\n"
        "---\n\n"
        f"# {title}\n\n"
        f"{body}\n"
    )
    filename = f"{date}-{title.lower().replace(' ', '-').replace(',', '').replace('—', '-')}.md"
    filename = "".join(c for c in filename if c.isalnum() or c in "-._")
    path = out_dir / filename
    path.write_text(text, encoding="utf-8")
    return path


def main(out_dir: Path = Path("scripts/sample_notes"), *, clean: bool = False) -> None:
    if clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for note in NOTES:
        write_note(
            out_dir,
            date=note["date"],
            title=note["title"],
            tags=note["tags"],
            body=note["body"],
        )
    print(f"Wrote {len(NOTES)} notes to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("scripts/sample_notes"))
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    main(args.out, clean=args.clean)
