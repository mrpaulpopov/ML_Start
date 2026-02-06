## Data Head

| index | text                                             | spam |
|------:|--------------------------------------------------|------|
| 0 | Subject: naturally irresistible your corporate... | 1 |
| 1 | Subject: the stock trading gunslinger fanny i...   | 1 |
| 2 | Subject: unbelievable new homes made easy im ...  | 1 |
| 3 | Subject: 4 color printing special request add...  | 1 |
| 4 | Subject: do not have money , get software cds ... | 1 |

---

## Pre-processed Data

| index | text                                             | words |
|------:|--------------------------------------------------|-------|
| 0 | Subject: naturally irresistible your corporate... | [content, statlonery, efforts, shouldn, benefi...] |
| 1 | Subject: the stock trading gunslinger fanny i...   | [group, not, pirogue, trading, esmark, continu...] |
| 2 | Subject: unbelievable new homes made easy im ...  | [-, pre, no, 72, visit, limited, we, loan, mad...] |
| 3 | Subject: 4 color printing special request add...  | [91706, now, pdf, 8090, (, -, @, ca, mail, for...] |
| 4 | Subject: do not have money , get software cds ... | [not, tradgedies, with, be, get, best, do, old...] |


---

## Dataset Statistics

- **Number of emails:** 5728  
- **Number of spam emails:** 1368  

**Prior probability of spam:**

```math
P(\text{spam}) = \frac{1368}{5728} = 0.2388268156424581
```
---

## Spam and Ham Counters (Laplace smoothing applied)

- **"Lottery":** `{'spam': 9, 'ham': 1}`
- **"Sale":** `{'spam': 39, 'ham': 42}`
- **"Mum":** `{'spam': 1, 'ham': 4}`

---

## Bayes Probabilities (Single Word)
```math
P(\text{spam} \mid \text{"lottery"}) = 0.9 \\
```

```math
P(\text{spam} \mid \text{"sale"}) = 0.48148148148148145 \\
```

---

## User Input

**Input word:** `lottery sale`

**Probability that the email is spam:**  
`0.9638144992048691` ✅

**Input word:** `meeting postponed`

**Probability that the email is spam:**  
`0.007179588721703268` ❌

**Input words:** `qweqwewqwe` *(not in the word list)*

**Probability that the email is spam:**  
`0.2388268156424581` 🟡 *(prior probability)*