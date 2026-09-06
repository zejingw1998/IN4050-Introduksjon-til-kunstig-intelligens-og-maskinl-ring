# IN3050 Lessons

This project contains Jupyter-based lessons for IN3050/IN4050 topics.

## Structure
```
.
├── Lecture_01_Linear_Algebra               # Relevant resources to each lecture.
│   ├── 01_theory_linear_algebra.ipynb
│   └── 02_practice_linear_algebra.ipynb
├── Lecture_02_Optimization_Search
├── Lecture_03_Supervised_Learning_KNN
├── requirements.txt                        # Python dependencies.
├── setup_env.bat                           # Windows environment setup.
└── setup_env.sh                            # macOS/Linux/Git Bash environment setup.
```

Each lecture folder should contain only files that belong directly to that lecture. Practice notebooks may include encoded answer cells next to the exercises. 

## Setup

### Python
Make sure you have python installed.

```
python --version
```
If python is not present, you can download it through their [website](https://www.python.org/downloads/).

**Note:** macOS does **not** have an official Python in the App Store. Check what *python3* points to:

```sh
which python3
```
We want it pointing to somewhere like.
- `/usr/local/bin/python3`
- `/opt/homebrew/bin/python3`

If `which python3` shows `/usr/bin/python3`, refer back to their website or homebrew:
   - [python.org/macos](https://www.python.org/downloads/macos/)
   - Homebrew: `brew install python3` (if you're familiar with Homebrew)

Open a new terminal window after installing, then re-check with `which python3`.

### Virtual environment
*Run this once to create a virtual environment and registers the Jupyter kernel used below, whether you work in the terminal or in VS Code.*

Windows:

```bat
setup_env.bat
```

macOS/Linux/Git Bash:

```sh
sh setup_env.sh
```

Optional - Start Jupyter Lab from the project root

```sh
jupyter lab
```

### VSCode
*If you wish to run jupyter notebook in VSCode*

Windows:

- [website](https://code.visualstudio.com/)

**Note:** When going through the installer, check the "add to PATH" option.

macOS:

Download it from their [website](https://code.visualstudio.com/), unzip it and drag `Visual Studio Code.app` into `/Applications` on your computer (this might require administrative permissions).

Another option is Homebrew:
```
brew install --cask visual-studio-code
```

Linux:
```sh
sudo snap install --classic code
```
Or download the `.deb`/`.rpm` from [their website](https://code.visualstudio.com/) and install it manually.

**Extensions**
- Jupyter: required to get jupyter notebook running.
- Pylance: provides autocomplete.
- Ruff: code formatting and linting.

**Select kernel**

After opening the jupyter notebook file, look towards the top right and select the instance you ran the setup script with via the "Select kernel" window.
