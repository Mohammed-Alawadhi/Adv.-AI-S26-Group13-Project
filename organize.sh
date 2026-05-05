#!/usr/bin/env bash
# ================================================================
#  organize.sh — adapted for your current folder state
#
#  Run from the project root:
#      cd /Users/error/Downloads/Adv.-AI-S26-Group13-Project
#      chmod +x organize.sh
#      ./organize.sh
# ================================================================

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

step() { echo -e "${BLUE}==> $1${NC}"; }
info() { echo -e "${GREEN}    ✓ $1${NC}"; }
warn() { echo -e "${YELLOW}    ⚠ $1${NC}"; }
fail() { echo -e "${RED}    ERROR: $1${NC}"; exit 1; }

ROOT="$(pwd)"
echo -e "${GREEN}Working in: ${ROOT}${NC}"
echo ""

[ -d "tdmpc2-highway" ] || fail "Folder 'tdmpc2-highway' not found. Are you in the right directory?"

# ================================================================
step "Step 1/7: Cleanup junk"

find . -name ".DS_Store" -delete 2>/dev/null || true
info "Removed .DS_Store files"

find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
info "Removed __pycache__/ directories"

# Remove the old combined script (we use organize.sh + push.sh now)
[ -f "organize_and_push.sh" ] && rm "organize_and_push.sh" && info "Removed old organize_and_push.sh"

# Empty Adv AI/ folder
if [ -d "Adv AI" ]; then
    if [ -z "$(ls -A "Adv AI" 2>/dev/null)" ]; then
        rmdir "Adv AI"
        info "Removed empty 'Adv AI/' folder"
    else
        warn "'Adv AI/' has files:"
        ls "Adv AI/" | sed 's/^/        /'
    fi
fi

# ================================================================
step "Step 2/7: Cleanup tdmpc2-highway/ junk and stale data"

[ -d "tdmpc2-highway/third_party" ] && rm -rf "tdmpc2-highway/third_party" && info "Removed third_party/"
[ -d "tdmpc2-highway/wandb" ]       && rm -rf "tdmpc2-highway/wandb"       && info "Removed wandb/ (auto-generated)"

for d in "tdmpc2-highway/logs" "tdmpc2-highway/configs" "tdmpc2-highway/scripts" "tdmpc2-highway/videos" "tdmpc2-highway/notebooks"; do
    [ -d "$d" ] && [ -z "$(ls -A "$d" 2>/dev/null)" ] && rmdir "$d" && info "Removed empty $d"
done

# Stale checkpoints
for ckpt in "tdmpc2-highway/checkpoints/sac_highway-v0_seed99" \
            "tdmpc2-highway/checkpoints/tdmpc2_highway-v0_seed99" \
            "tdmpc2-highway/checkpoints/tdmpc2_size5_h3_seed1" \
            "tdmpc2-highway/checkpoints/tdmpc2_size5_h5_seed99"; do
    if [ -d "$ckpt" ]; then
        rm -rf "$ckpt"
        info "Removed stale checkpoint: $(basename "$ckpt")"
    fi
done

# Debug intermediate JSONs
[ -f "tdmpc2-highway/results/eval_results_tdmpc2_v2.json" ] && \
    rm "tdmpc2-highway/results/eval_results_tdmpc2_v2.json" && \
    info "Removed eval_results_tdmpc2_v2.json (debug)"
[ -f "tdmpc2-highway/results/ablation_eval_mode.json" ] && \
    rm "tdmpc2-highway/results/ablation_eval_mode.json" && \
    info "Removed ablation_eval_mode.json (debug)"

# ================================================================
step "Step 3/7: Set up notebooks/ folder"

mkdir -p notebooks

# Drop the old evaluate notebook, rename the fixed one
if [ -f "03_evaluate.ipynb" ] && [ -f "03_evaluate_fixed.ipynb" ]; then
    rm "03_evaluate.ipynb"
    info "Removed older 03_evaluate.ipynb (95 KB)"
    mv "03_evaluate_fixed.ipynb" "notebooks/03_evaluate.ipynb"
    info "Moved & renamed 03_evaluate_fixed.ipynb → notebooks/03_evaluate.ipynb"
fi

# Move other notebooks
for nb in "01_train_sac.ipynb" "02_train_tdmpc2.ipynb" "02b_retrain_tdmpc2_size5.ipynb"; do
    if [ -f "$nb" ]; then
        mv "$nb" "notebooks/$nb"
        info "Moved $nb → notebooks/"
    fi
done

# ================================================================
step "Step 4/7: Set up docs/ folder and move PDFs"

mkdir -p docs
mkdir -p docs/paper

# Survey paper at root → docs/survey.pdf
if [ -f "A_Comparative_Survey_of_Model_Based_and_Model_Free_Reinforcement_Learning.pdf" ]; then
    mv "A_Comparative_Survey_of_Model_Based_and_Model_Free_Reinforcement_Learning.pdf" "docs/survey.pdf"
    info "Moved survey paper → docs/survey.pdf"
fi

# Course handout
if [ -f "survey paper   research project (1).pdf" ]; then
    mv "survey paper   research project (1).pdf" "docs/handout.pdf"
    info "Moved course handout → docs/handout.pdf"
fi

# Compiled paper PDF at root → docs/paper.pdf
if [ -f "Model_Based_vs__Model_Free_Reinforcement_Learning_for_Autonomous_Driving.pdf" ]; then
    mv "Model_Based_vs__Model_Free_Reinforcement_Learning_for_Autonomous_Driving.pdf" "docs/paper.pdf"
    info "Moved compiled paper → docs/paper.pdf"
fi

# Deadlines doc
if [ -f "Deadlines.docx" ]; then
    mv "Deadlines.docx" "docs/Deadlines.docx"
    info "Moved Deadlines.docx → docs/"
fi

# Optional: prompt for main.tex / references.bib (these are usually in Overleaf)
echo ""
echo -e "${YELLOW}    Optional: paths to LaTeX source files. Press ENTER to skip.${NC}"
read -p "    main.tex path: " TEX_PATH
TEX_PATH="${TEX_PATH//\'/}"
TEX_PATH="$(echo "$TEX_PATH" | sed 's/\\ / /g' | xargs 2>/dev/null || echo "$TEX_PATH")"
if [ -n "$TEX_PATH" ] && [ -f "$TEX_PATH" ]; then
    cp "$TEX_PATH" "docs/paper/main.tex"
    info "Copied main.tex → docs/paper/main.tex"
fi

read -p "    references.bib path: " BIB_PATH
BIB_PATH="${BIB_PATH//\'/}"
BIB_PATH="$(echo "$BIB_PATH" | sed 's/\\ / /g' | xargs 2>/dev/null || echo "$BIB_PATH")"
if [ -n "$BIB_PATH" ] && [ -f "$BIB_PATH" ]; then
    cp "$BIB_PATH" "docs/paper/references.bib"
    info "Copied references.bib → docs/paper/references.bib"
fi

# Clean up empty docs/paper if nothing was added
if [ -d "docs/paper" ] && [ -z "$(ls -A docs/paper 2>/dev/null)" ]; then
    rmdir "docs/paper"
    warn "docs/paper/ left empty (you can add main.tex and references.bib later)"
fi

# ================================================================
step "Step 5/7: Flatten tdmpc2-highway/* up to repo root"

cd tdmpc2-highway
shopt -s dotglob 2>/dev/null || true
for item in *; do
    [ -e "$item" ] || continue
    if [ -e "../$item" ]; then
        warn "Skipping $item (already exists at root)"
    else
        mv "$item" "../$item"
        info "Moved $item → repo root"
    fi
done
cd ..

# Remove empty tdmpc2-highway/
if [ -d "tdmpc2-highway" ] && [ -z "$(ls -A tdmpc2-highway 2>/dev/null)" ]; then
    rmdir tdmpc2-highway
    info "Removed empty 'tdmpc2-highway/' folder"
fi

# ================================================================
step "Step 6/7: Optional wandb data import"

read -p "    wandb-metadata.json path (ENTER to skip): " META_PATH
META_PATH="${META_PATH//\'/}"
META_PATH="$(echo "$META_PATH" | sed 's/\\ / /g' | xargs 2>/dev/null || echo "$META_PATH")"
if [ -n "$META_PATH" ] && [ -f "$META_PATH" ]; then
    mkdir -p results/wandb
    cp "$META_PATH" "results/wandb/wandb-metadata.json"
    info "Copied wandb-metadata.json"
fi

read -p "    Folder containing wandb_export_*.csv files (ENTER to skip): " CSV_DIR
CSV_DIR="${CSV_DIR//\'/}"
CSV_DIR="$(echo "$CSV_DIR" | sed 's/\\ / /g' | xargs 2>/dev/null || echo "$CSV_DIR")"
if [ -n "$CSV_DIR" ] && [ -d "$CSV_DIR" ]; then
    mkdir -p results/wandb/size5_run
    if cp "$CSV_DIR"/wandb_export_*.csv results/wandb/size5_run/ 2>/dev/null; then
        info "Copied wandb CSVs"
    else
        warn "No wandb_export_*.csv files in $CSV_DIR"
        rmdir results/wandb/size5_run 2>/dev/null || true
    fi
fi

# ================================================================
step "Step 7/7: Verify presence of repo files"

for f in README.md LICENSE requirements.txt .gitignore; do
    if [ -f "./$f" ]; then
        info "$f present"
    else
        warn "$f MISSING — copy from chat downloads before push"
    fi
done

echo ""
echo -e "${GREEN}================================================================${NC}"
echo -e "${GREEN}  ✓ Reorganization complete${NC}"
echo -e "${GREEN}================================================================${NC}"
echo ""
echo "Verify with:"
echo "  ls -la"
echo "  ls -la notebooks/  docs/  checkpoints/  results/"
echo "  du -sh ."
