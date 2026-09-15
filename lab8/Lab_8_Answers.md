# Lab 8 – Genetic Algorithms & AI — Answers

## Problem 1 — Paper study
Read: V. Hristakeva & D. Shrestha, "Solving the 0/1 Knapsack Problem with Genetic Algorithms" (MICS 2004). Key ideas used below: 0/1 bit-string encoding, penalty for exceeding capacity, roulette-wheel selection, single-point crossover, bit-flip mutation.

## Problem 2 — Backing up 5000 MP3s onto CDs with a GA

**Encoding.** Same as the 0/1 knapsack problem: a chromosome is a bit string of length 5000, one bit per MP3. Bit = 1 means the MP3 is placed on the current CD, 0 means it is not. The "weight" of each item is the MP3's file size; the knapsack capacity is the CD capacity (700 MB).

**Fitness function.** The total size of the selected MP3s — the closer to 700 MB the better, so each CD is filled as completely as possible. Chromosomes whose total size exceeds the CD capacity are invalid and are penalized (or repaired by flipping 1s to 0s until they fit).

**Genetic operators.**
- *Selection:* roulette-wheel (fitness-proportionate) or tournament selection, with elitism so the best packing is never lost.
- *Crossover:* single-point crossover on the bit strings.
- *Mutation:* flip a random bit with a small probability (include/exclude one MP3).

**Handling multiple CDs.** Run the GA iteratively: use it to fill CD #1 as completely as possible, remove those MP3s from the pool, then run the GA again on the remaining files for CD #2, and repeat until no files remain. Each run is a single-knapsack problem, exactly like the paper. (Alternative: encode each gene as an integer CD number 1..k per MP3 and minimize the number of CDs plus wasted space, but the iterative approach is simpler and reuses the paper's GA directly.)

## Problem 3 — PEAS descriptions

PEAS = **P**erformance measure (how success is judged), **E**nvironment (where the agent operates), **A**ctuators (hardware/means to act), **S**ensors (means to perceive).

### Robot soccer player
| | |
|---|---|
| **P** | Winning the game, goals scored, goals conceded, ball possession |
| **E** | Field, ball, own team, opposing team, referee/rules |
| **A** | Legs/motors for running, kicking mechanism, body for ball control |
| **S** | Camera, touch/proximity sensors, accelerometer/gyroscope, communication link with teammates |

### Internet book-shopping agent
| | |
|---|---|
| **P** | Obtaining the requested books at minimum price, quality/relevance of results, purchase speed |
| **E** | The internet: online bookstore websites, vendors, prices, reviews |
| **A** | Display results to the user, fill in forms, follow links, place orders |
| **S** | Web pages (HTML content), user's requests/keyboard input |

### Autonomous Mars rover
| | |
|---|---|
| **P** | Terrain explored, scientific data/samples collected, power conserved, rover kept undamaged |
| **E** | Martian surface: rocks, craters, dust storms, extreme temperatures; no humans nearby |
| **A** | Wheels/motors, robotic arm, drill, sample containers, radio transmitter |
| **S** | Cameras, terrain/obstacle sensors, spectrometers, temperature sensors, GPS-like localization, radio receiver |

### Mathematician's theorem-proving assistant
| | |
|---|---|
| **P** | Correctness of proofs, number of theorems proved, proof length/elegance, time taken |
| **E** | Axioms, definitions, previously proved theorems, the conjecture to prove |
| **A** | Display/print proof steps, suggest lemmas, flag errors |
| **S** | Keyboard input — formulas, conjectures, and commands from the mathematician |

## Problem 4 — Natural-language Medical Assistant Agent

**Agent type:** utility-based agent with an internal state/memory (it must track conversation and patient context, reason about consequences, and trade off risk vs. usefulness — a pure reflex agent is not safe enough for medicine).

**Architecture:**

```
          ENVIRONMENT (patient/user, medical records, drug databases, human doctors)
                 │  natural-language symptoms & questions
                 ▼
        ┌── SENSORS: text/speech input → NLP parser ──────────────┐
        │                                                          │
        │  State tracker (context/memory):                        │
        │    "What is the patient's situation now?"               │
        │    conversation history + patient history               │
        │                                                          │
        │  Medical knowledge base:                                 │
        │    "How the world works" — diseases, drugs, guidelines  │
        │                                                          │
        │  Reasoning engine + goals & utility:                     │
        │    "What happens if I suggest X? Is it safe/useful?"    │
        │                                                          │
        └── ACTUATORS: text/speech reply, urgent alerts, ──────────┘
            appointment booking, escalation to a human doctor
                 │
                 ▼
          back to ENVIRONMENT
```

**Goals:** correctly identify the patient's problem; give safe, evidence-based advice; escalate emergencies to a human doctor; never act beyond its authority (no prescriptions).

**Utility function:** maximize diagnostic accuracy and patient safety; minimize response time and unnecessary escalations; weight "do no harm" heavily — a confident wrong answer is worse than recommending a doctor visit.

## Problem 5 — Depth-First Search order

Tree: 1 → {2, 3, 4}; 2 → {5, 6}; 5 → {11}; 11 → {13, 14}; 14 → {17}; 3 → {7, 8}; 7 → {12}; 12 → {15, 16}; 4 → {9, 10}.

DFS visits each node when it is first reached, starting at the root and exploring each branch as deep as possible (left to right):

**1, 2, 5, 11, 13, 14, 17, 6, 3, 7, 12, 15, 16, 8, 4, 9, 10**

## Problem 6 — Travelling Salesman Problem with a GA

**GA design:**
- *Encoding:* a chromosome is a **permutation** of all cities (the visiting order). Unlike knapsack, bits won't work — every city must appear exactly once.
- *Fitness:* 1 / total route distance (the route wraps back to the starting city), so shorter routes have higher fitness.
- *Selection:* elitism + roulette-wheel selection.
- *Crossover:* **ordered crossover** — copy a random slice from parent 1, fill the remaining positions with the cities in parent 2's order. Plain single-point crossover would duplicate/lose cities and produce invalid routes.
- *Mutation:* **swap mutation** — exchange two cities in the route (keeps the permutation valid).

**Implementation:** see `lab8_tsp_ga.py` (uses numpy, pandas, random, operator).

Sample run (25 random cities, population 100, elite size 20, mutation rate 0.01, 500 generations):

```
Initial distance: 2000.15
Final distance:   793.64
```
