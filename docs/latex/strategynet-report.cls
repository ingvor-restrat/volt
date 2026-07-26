\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{strategynet-report}
  [2026/07/24 v0.4 StrategyNet.ai technical report]

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax
\LoadClass[10pt,twoside]{article}

% TikZ loads xcolor transitively. Declare the palette's table requirement
% before TikZ so the later palette import does not trigger an option clash.
\PassOptionsToPackage{table}{xcolor}

% Match the compact research-note typography used by BRCentralRiskBook:
% Palatino for technical prose and Helvetica for the publication hierarchy.
% TeX Gyre Pagella and Heros are the OpenType equivalents under LuaLaTeX.
\RequirePackage{iftex}
\ifPDFTeX
  \RequirePackage[T1]{fontenc}
  \RequirePackage[utf8]{inputenc}
\fi
\RequirePackage[english]{babel}
\ifPDFTeX
  \RequirePackage{mathpazo}
  \RequirePackage[scaled]{helvet}
  \RequirePackage[scaled=0.88]{inconsolata}
\else
  \RequirePackage{fontspec}
  \RequirePackage{unicode-math}
  \setmainfont{TeX Gyre Pagella}
  \setsansfont[Scale=MatchLowercase]{TeX Gyre Heros}
  \setmathfont{TeX Gyre Pagella Math}
  \setmonofont[Scale=0.88]{Inconsolatazi4}
\fi
\RequirePackage{microtype}

\RequirePackage[
  letterpaper,
  inner=0.70in,
  outer=0.64in,
  top=0.68in,
  bottom=0.72in,
  headheight=19pt,
  headsep=7pt,
  footskip=24pt
]{geometry}

\RequirePackage{amsmath}
\ifPDFTeX
  \RequirePackage{amssymb}
\fi
\RequirePackage{graphicx}
\RequirePackage{booktabs}
\RequirePackage{array}
\RequirePackage{tabularx}
\RequirePackage{multirow}
\RequirePackage{multicol}
\RequirePackage{enumitem}
\RequirePackage{ragged2e}
\RequirePackage{needspace}
\RequirePackage{listings}
\RequirePackage{tikz}
\usetikzlibrary{arrows.meta,calc,fit,positioning,shapes.geometric}
\RequirePackage[most]{tcolorbox}
\RequirePackage{caption}
\RequirePackage{pdflscape}
\RequirePackage{fancyhdr}
\RequirePackage{lastpage}
\RequirePackage{titlesec}
\RequirePackage{tocloft}
\RequirePackage{etoolbox}
\RequirePackage{xparse}
\RequirePackage{xurl}

\graphicspath{{assets/}}

% The class is normally found through \input@path.  The fallback keeps builds
% from the repository root simple without installing a private TeX package.
\IfFileExists{strategynet-palette.sty}
  {\RequirePackage{strategynet-palette}}
  {\RequirePackage{docs/latex/strategynet/strategynet-palette}}

\RequirePackage{hyperref}
\hypersetup{
  colorlinks=true,
  linkcolor=SNViolet,
  citecolor=SNSignalDark,
  urlcolor=SNLink,
  bookmarksopen=true,
  bookmarksnumbered=true,
  pdfdisplaydoctitle=true
}
\urlstyle{same}

\color{SNInk}
\pagecolor{SNPaper}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.38em plus 0.08em minus 0.05em}
\setlength{\columnsep}{18pt}
\setlength{\columnseprule}{0.25pt}
\setlength{\multicolsep}{0.55em plus 0.15em minus 0.10em}
\setlength{\premulticols}{0.45em}
\setlength{\postmulticols}{0.45em}
\setlength{\emergencystretch}{1.2em}
\clubpenalty=9000
\widowpenalty=9000
\displaywidowpenalty=9000
\raggedbottom

\setlist[itemize]{
  leftmargin=1.45em,
  itemsep=0.16em,
  parsep=0pt,
  topsep=0.30em,
  partopsep=0pt
}
\setlist[enumerate]{
  leftmargin=1.65em,
  itemsep=0.18em,
  parsep=0pt,
  topsep=0.32em,
  partopsep=0pt
}
\renewcommand{\arraystretch}{1.12}
\arrayrulecolor{SNLine}

\captionsetup{
  font=small,
  labelfont={sf,bf,color=SNMidnight},
  textfont={color=SNSlate},
  labelsep=period,
  justification=raggedright,
  singlelinecheck=false,
  skip=4pt
}

\lstdefinestyle{sncode}{
  basicstyle=\ttfamily\small,
  keywordstyle=\color{SNViolet}\bfseries,
  commentstyle=\color{SNMuted}\itshape,
  stringstyle=\color{SNSignalDark},
  numbers=none,
  frame=single,
  rulecolor=\color{SNLine},
  backgroundcolor=\color{SNCodeBackground},
  framesep=5pt,
  xleftmargin=2pt,
  xrightmargin=2pt,
  breaklines=true,
  breakatwhitespace=false,
  showstringspaces=false,
  columns=fullflexible
}

% Compact hierarchy. Section numbers act as navigational markers instead of
% full-width bars, leaving more room for mixed one- and two-column layouts.
\titleformat{\part}[display]
  {\sffamily\bfseries\color{SNMidnight}}
  {\color{SNSignalDark}\fontsize{9}{10}\selectfont
   PART \MakeUppercase{\thepart}}
  {0.30em}
  {\fontsize{23}{25}\selectfont}
  [\vspace{0.35em}\color{SNLine}\titlerule]
\titlespacing*{\part}{0pt}{0pt}{0.95em}

\titleformat{\section}[hang]
  {\sffamily\bfseries\color{SNMidnight}\fontsize{15}{17}\selectfont}
  {\colorbox{SNSignal}{%
    \makebox[1.70em][c]{\color{white}\fontsize{9}{10}\selectfont\thesection}}}
  {0.58em}
  {}
\titlespacing*{\section}{0pt}{1.35ex plus 0.35ex minus 0.20ex}
  {0.52ex plus 0.10ex}

\titleformat{\subsection}[hang]
  {\sffamily\bfseries\color{SNMidnight}\fontsize{11}{13}\selectfont}
  {\textcolor{SNSignalDark}{\thesubsection}}
  {0.52em}
  {}
\titlespacing*{\subsection}{0pt}{1.05ex plus 0.25ex minus 0.15ex}
  {0.32ex}

\titleformat{\subsubsection}[hang]
  {\sffamily\bfseries\color{SNSlate}\fontsize{9.5}{11}\selectfont}
  {\textcolor{SNViolet}{\thesubsubsection}}
  {0.48em}
  {}
\titlespacing*{\subsubsection}{0pt}{0.85ex plus 0.20ex minus 0.10ex}
  {0.20ex}

\titleformat{\paragraph}[runin]
  {\sffamily\bfseries\color{SNMidnight}}
  {}
  {0pt}
  {}
  [.\enspace]
\titlespacing*{\paragraph}{0pt}{0.65ex}{0.35em}

\pretocmd{\part}{\clearpage}{}{}
\pretocmd{\section}{\Needspace{9\baselineskip}}{}{}

% Publication metadata.
\makeatletter
\def\@snsubtitle{}
\def\@sndocumenttype{Technical report}
\def\@snversion{Working draft}
\def\@sndate{\today}
\def\@snstatus{Open-source publication}
\def\@snshorttitle{Volt Technical Report}
\def\@sncoverimage{assets/strategynet-cover-mountains-monochrome.jpg}

\newcommand{\SNSubtitle}[1]{\gdef\@snsubtitle{#1}}
\newcommand{\SNDocumentType}[1]{\gdef\@sndocumenttype{#1}}
\newcommand{\SNVersion}[1]{\gdef\@snversion{#1}}
\newcommand{\SNDate}[1]{\gdef\@sndate{#1}}
\newcommand{\SNStatus}[1]{\gdef\@snstatus{#1}}
\newcommand{\SNShortTitle}[1]{\gdef\@snshorttitle{#1}}
\newcommand{\SNCoverImage}[1]{\gdef\@sncoverimage{#1}}

\renewcommand{\maketitle}{%
  \thispagestyle{empty}%
  \begingroup
  \setlength{\parskip}{0pt}%
  \noindent
  \begin{tikzpicture}
    \path[use as bounding box] (0,0) rectangle (\textwidth,7.45cm);
    \begin{scope}
      \clip[rounded corners=2pt] (0,0) rectangle (\textwidth,7.45cm);
      \fill[SNMidnight] (0,0) rectangle (\textwidth,7.45cm);
      \IfFileExists{\@sncoverimage}{%
        \node[anchor=south west,inner sep=0pt] at (0,0)
          {\includegraphics[width=\textwidth]{\@sncoverimage}};
        \fill[SNMidnight,opacity=0.42]
          (0,0) rectangle (\textwidth,7.45cm);
      }{}
    \end{scope}
    \fill[SNSignal] (0,0) rectangle (\textwidth,0.13cm);

    \node[anchor=north west] at (0.48cm,6.98cm)
      {\fontsize{15}{17}\selectfont\SNwordmarkondark};
    \node[
      anchor=north west,
      text=white,
      text width=0.76\textwidth,
      align=left,
      font=\sffamily\bfseries\fontsize{27}{29}\selectfont
    ] at (0.48cm,5.72cm) {\@title};
    \node[
      anchor=north west,
      text=white!78,
      text width=0.78\textwidth,
      align=left,
      font=\sffamily\fontsize{12.5}{15.5}\selectfont
    ] at (0.50cm,3.66cm) {\@snsubtitle};
    \node[
      anchor=south west,
      text=white!76,
      font=\sffamily\bfseries\fontsize{8.5}{10}\selectfont
    ] at (0.50cm,0.56cm) {%
      \MakeUppercase{\@sndocumenttype}\enspace
      \textcolor{white!42}{\textbullet}\enspace
      \MakeUppercase{\@snversion}};
  \end{tikzpicture}

  \vspace{0.42em}
  \begin{tabularx}{\textwidth}{@{}
    >{\sffamily\small\bfseries\color{SNMidnight}}X
    >{\sffamily\small\color{SNSlate}\raggedleft\arraybackslash}p{0.50\textwidth}@{}}
    \@author &
    \textcolor{SNSignalDark}{\@snstatus}\enspace
    \textcolor{SNLine}{\textbullet}\enspace
    \@sndate
  \end{tabularx}
  \vspace{0.28em}
  \color{SNLine}\hrule
  \endgroup
}

\renewenvironment{abstract}
  {\begin{tcolorbox}[
      enhanced,
      breakable,
      colback=SNMist,
      colframe=SNLine,
      boxrule=0.45pt,
      arc=1.5pt,
      left=8pt,
      right=8pt,
      top=6pt,
      bottom=7pt,
      borderline west={2.2pt}{0pt}{SNSignal},
      before skip=0.75em,
      after skip=0.45em
    ]
    {\sffamily\bfseries\fontsize{8.5}{10}\selectfont
     \color{SNSignalDark}ABSTRACT}\par\smallskip
    \small\color{SNInk}}
  {\end{tcolorbox}}

\newcommand{\SNtableofcontents}{%
  \clearpage
  \thispagestyle{plain}%
  \begingroup
    \setlength{\columnsep}{24pt}%
    \setlength{\columnseprule}{0pt}%
    \begin{multicols}{2}
      \tableofcontents
    \end{multicols}
  \endgroup
  \clearpage
}

% Running identity.
\pagestyle{fancy}
\fancyhf{}
\fancyhead[LE,RO]{\sffamily\footnotesize\color{SNSlate}\@snshorttitle}
\fancyhead[LO,RE]{\sffamily\footnotesize\color{SNMuted}\nouppercase{\rightmark}}
\fancyfoot[LE,RO]{\sffamily\footnotesize\color{SNMidnight}\thepage}
\fancyfoot[LO,RE]{\sffamily\footnotesize\SNwordmark}
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0pt}
\renewcommand{\headrule}{%
  \hbox to\headwidth{%
    \color{SNSignalDark}\leaders\hrule height \headrulewidth\hfill}}
\renewcommand{\sectionmark}[1]{\markright{\thesection\enspace #1}}
\fancypagestyle{plain}{%
  \fancyhf{}
  \fancyfoot[LE,RO]{\sffamily\footnotesize\color{SNMidnight}\thepage}
  \fancyfoot[LO,RE]{\sffamily\footnotesize\SNwordmark}
  \renewcommand{\headrulewidth}{0pt}
}
\makeatother

% Compact, two-column table of contents.
\setcounter{tocdepth}{2}
\setcounter{secnumdepth}{3}
\renewcommand{\contentsname}{Contents}
\renewcommand{\cfttoctitlefont}{\sffamily\bfseries\color{SNMidnight}\Huge}
\renewcommand{\cftaftertoctitle}{\par\color{SNLine}\hrule\vspace{0.55em}}
\setlength{\cftbeforepartskip}{0.55em}
\setlength{\cftbeforesecskip}{0.10em}
\setlength{\cftbeforesubsecskip}{0pt}
\renewcommand{\cftpartfont}{\sffamily\bfseries\color{SNSignalDark}}
\renewcommand{\cftsecfont}{\sffamily\small\bfseries\color{SNMidnight}}
\renewcommand{\cftsubsecfont}{\sffamily\footnotesize\color{SNSlate}}
\renewcommand{\cftpartpagefont}{\sffamily\bfseries\color{SNSignalDark}}
\renewcommand{\cftsecpagefont}{\sffamily\small\color{SNSlate}}
\renewcommand{\cftsubsecpagefont}{\sffamily\footnotesize\color{SNMuted}}
\renewcommand{\cftpartleader}{\cftdotfill{\cftdotsep}}
\renewcommand{\cftsecleader}{\cftdotfill{\cftdotsep}}

% Explicit layout primitives.  These make column changes visible in source and
% avoid a class-level two-column mode that traps wide figures and equations.
\newenvironment{SNcolumns}
  {\begingroup
   \def\columnseprulecolor{\color{SNLine}}%
   \begin{multicols}{2}\raggedcolumns}
  {\end{multicols}\endgroup}

\newcommand{\SNcolumnbreak}{\columnbreak}

\newenvironment{SNlandscape}
  {\clearpage\begin{landscape}}
  {\end{landscape}\clearpage}

\newcommand{\SNsectionpage}{\clearpage}

\newtcolorbox{SNnote}[1][Note]{
  enhanced,
  breakable,
  colback=SNPaper,
  colframe=SNLine,
  boxrule=0.45pt,
  arc=1.5pt,
  borderline west={2pt}{0pt}{SNSignal},
  title={#1},
  fonttitle=\sffamily\bfseries\small,
  coltitle=SNSignalDark,
  left=7pt,
  right=7pt,
  top=4pt,
  bottom=5pt,
  before skip=0.55em,
  after skip=0.55em
}

\newtcolorbox{SNthesis}[1][Core thesis]{
  enhanced,
  breakable,
  colback=SNMist,
  colframe=SNLine,
  boxrule=0.45pt,
  arc=2pt,
  borderline west={2.2pt}{0pt}{SNMuted},
  title={#1},
  fonttitle=\sffamily\bfseries\small,
  coltitle=SNSlate,
  coltext=SNInk,
  left=9pt,
  right=9pt,
  top=7pt,
  bottom=8pt,
  before skip=0.70em,
  after skip=0.70em
}

\newenvironment{tradernote}
  {\begin{SNnote}[Trader translation]}
  {\end{SNnote}}

\newcommand{\SNpill}[2][SNSignal]{%
  \tikz[baseline=(sn.base)]\node[
    rounded corners=3pt,
    fill=#1!12,
    draw=#1!45,
    inner xsep=5pt,
    inner ysep=2pt,
    text=#1!70!black,
    font=\sffamily\bfseries\scriptsize
  ] (sn) {#2};%
}

\newenvironment{SNcompactitem}
  {\begin{itemize}[itemsep=0.08em,topsep=0.18em]}
  {\end{itemize}}

\newenvironment{SNcompactenum}
  {\begin{enumerate}[itemsep=0.08em,topsep=0.18em]}
  {\end{enumerate}}

\endinput
