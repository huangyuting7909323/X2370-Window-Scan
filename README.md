\# README



\## Description



This dataset contains the paper, source code, and figure for the study:



\*\*Automatic Identification of Parameter Windows for Stable Solutions of a Nonlinear Scalar Field Equation: A Case Study of X(2370)\*\*



by Yuting Huang (ORCID: 0009-0000-6346-2905)



\## Files



\- `Automatic\_Identification\_of\_Parameter\_Windows.pdf` — paper (full text)

\- `window\_scan.py` — Python source code for parameter window identification

\- `parameter\_mass\_map.png` — parameter-mass map (Fig. 1 in the paper)

\- `README.md` — this file



\## Requirements



\- Python 3.14.4

\- numpy 2.4.4

\- scipy 1.17.1

\- matplotlib 3.10.8



Install with:



&#x20;   pip install numpy scipy matplotlib



\## Usage



&#x20;   python window\_scan.py



\## Runtime



Approximately 2 seconds per single solve, and about 30 minutes for a complete window scan (30×30 grid) on a conventional personal laptop.



\## Output



The program prints the window boundary and stable-window limits, and generates the parameter-mass map `parameter\_mass\_map.png`.



\## Contact



Yuting Huang

ORCID: 0009-0000-6346-2905

