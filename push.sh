#!/usr/bin/env bash
# ================================================================
#  push.sh — Initialize git, commit everything, and push to GitHub.
#            Reads PAT directly from the user (not interactively from git),
#            then embeds it in the remote URL just for this push.
#
#  Run from the project root:
#      cd /Users/error/Downloads/Adv.-AI-S26-Group13-Project
#      chmod +x push.sh
#      ./push.sh
# ================================================================

set -euo pipefail

# Repo config
GH_USER="Mohammed-Alawadhi"
GH_REPO="Adv.-AI-S26-Group13-Project"
DEFAULT_BRANCH="main"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

step() { echo -e "${BLUE}==> $1${NC}"; }
info() { echo -e "${GREEN}    ✓ $1${NC}"; }
warn() { echo -e "${YELLOW}    ⚠ $1${NC}"; }
fail() { echo -e "${RED}    ✗ ERROR: $1${NC}"; exit 1; }

ROOT="$(pwd)"
echo -e "${GREEN}Working in: ${ROOT}${NC}"
echo ""

# ================================================================
step "Pre-flight: cleanup any stray .DS_Store"
find . -name ".DS_Store" -delete 2>/dev/null || true
info "Done"

# ================================================================
step "Pre-flight: verify required files"
MISSING=0
for f in README.md LICENSE requirements.txt .gitignore; do
    if [ -f "./$f" ]; then
        info "$f"
    else
        warn "$f MISSING — push aborted"
        MISSING=1
    fi
done
[ "$MISSING" -eq 1 ] && fail "Required files missing. Add them and re-run."

# Show what we're about to push (size sanity check)
echo ""
step "Repo size summary"
du -sh */ 2>/dev/null | sort -h
echo "    Total: $(du -sh . | cut -f1)"

# ================================================================
step "Initializing git"

if [ -d ".git" ]; then
    warn ".git/ already exists — skipping git init"
else
    git init -q -b "$DEFAULT_BRANCH"
    info "Initialized git on branch '$DEFAULT_BRANCH'"
fi

# Check git identity
GIT_USER_GLOBAL=$(git config user.name 2>/dev/null || echo "")
GIT_EMAIL_GLOBAL=$(git config user.email 2>/dev/null || echo "")
if [ -z "$GIT_USER_GLOBAL" ] || [ -z "$GIT_EMAIL_GLOBAL" ]; then
    echo ""
    warn "Git identity not set. Configuring locally for this repo."
    read -p "    Your name (e.g., 'Mohammed Alawadhi'): " GIT_USER_INPUT
    read -p "    Your email (matching your GitHub account): " GIT_EMAIL_INPUT
    git config user.name "$GIT_USER_INPUT"
    git config user.email "$GIT_EMAIL_INPUT"
    info "Set local git identity"
else
    info "Git identity: $GIT_USER_GLOBAL <$GIT_EMAIL_GLOBAL>"
fi

# ================================================================
step "Staging files (respecting .gitignore)"

git add -A
ADDED_FILES=$(git diff --cached --name-only | wc -l | tr -d ' ')
info "Staged $ADDED_FILES files"
echo ""
echo "    First 20 staged files:"
git diff --cached --name-only | head -20 | sed 's/^/      /'
TOTAL_STAGED=$(git diff --cached --name-only | wc -l | tr -d ' ')
[ "$TOTAL_STAGED" -gt 20 ] && echo "      ... and $((TOTAL_STAGED - 20)) more"

# ================================================================
step "Committing"

if git diff --cached --quiet; then
    warn "Nothing to commit (working tree clean)"
else
    git commit -q -m "Initial commit: MLR 555 Group 13 research project

Project: Model-Based vs. Model-Free Reinforcement Learning for Autonomous Driving
A comparative study on highway-env using TD-MPC2 (ICLR 2024) and SAC.

Authors:
- Nihal Abdul Naseer (b00112155)
- Mohammed Alawadhi (b00108492)
- Abdulhafedh Al-Zubedi (b00112488)

Contents:
- 4 Colab notebooks (training + evaluation + size=5 robustness check)
- 7 trained checkpoints (3 SAC + 3 TD-MPC2 size=1 + 1 TD-MPC2 size=5)
- Full evaluation grid: 540 episodes (2 algos x 3 seeds x 3 envs x 30 eps)
- 5 paper figures (PDF + PNG)
- World-model latent-space dynamics probes (V1 + V2)
- Statistical analysis (Cohen's d, paired t-test, Mann-Whitney U)
- W&B run history for all 9 logged training runs
- Compiled paper PDF (IEEE journal format)
- Prior survey paper PDF

Headline finding:
TD-MPC2 retains 85% of training reward under cross-action-space zero-shot
transfer vs. only 21% for SAC (4.0x larger transfer factor). On roundabout-v0
specifically, TD-MPC2 achieves 53.3% success vs. SAC's 43.3% (Cohen's d=0.88,
large effect; every TD-MPC2 seed exceeds SAC mean)."
    info "Made initial commit"
fi

# ================================================================
step "Reading GitHub Personal Access Token"

echo ""
echo -e "${YELLOW}    Paste your GitHub Personal Access Token (PAT).${NC}"
echo -e "${YELLOW}    Generate one at: https://github.com/settings/tokens (scope: 'repo')${NC}"
echo -e "${YELLOW}    The token will NOT be visible as you paste — that's normal.${NC}"
echo ""

# Read PAT silently (no echo)
read -s -p "    PAT: " GH_TOKEN
echo ""
echo ""

if [ -z "$GH_TOKEN" ]; then
    fail "No PAT provided. Aborted."
fi

# Quick sanity check on PAT format
if [[ ! "$GH_TOKEN" =~ ^(ghp_|github_pat_) ]]; then
    warn "Your PAT does not start with 'ghp_' or 'github_pat_' — that's unusual."
    warn "Continuing anyway, but if push fails, double-check the token."
    echo ""
fi

# Build authenticated URL (only used for this single push)
AUTH_URL="https://${GH_USER}:${GH_TOKEN}@github.com/${GH_USER}/${GH_REPO}.git"
SAFE_URL="https://github.com/${GH_USER}/${GH_REPO}.git"

# ================================================================
step "Configuring remote"

if git remote get-url origin >/dev/null 2>&1; then
    git remote set-url origin "$SAFE_URL"
    info "Updated remote to: $SAFE_URL"
else
    git remote add origin "$SAFE_URL"
    info "Added remote: $SAFE_URL"
fi

# ================================================================
step "Pushing to GitHub"

echo "    (Using authenticated URL with embedded PAT for this push only.)"
echo ""

# Push using the authenticated URL directly (overrides 'origin' for this command)
if git push "$AUTH_URL" "$DEFAULT_BRANCH"; then
    info "Push succeeded!"
else
    fail "Push failed. Check the error above. Common causes:
       - Invalid PAT (wrong characters, expired, or wrong scope)
       - Repo isn't actually empty (try 'git pull origin main --rebase --allow-unrelated-histories' then re-run)
       - Network issue"
fi

# Set upstream tracking on the safe (token-free) remote
git branch --set-upstream-to=origin/"$DEFAULT_BRANCH" "$DEFAULT_BRANCH" 2>/dev/null || true

# Make sure the saved remote URL has NO embedded token
git remote set-url origin "$SAFE_URL"
info "Cleaned token from saved remote URL (next pushes will use credential helper or prompt)"

# ================================================================
echo ""
echo -e "${GREEN}================================================================${NC}"
echo -e "${GREEN}  ✓✓✓ Push complete!${NC}"
echo -e "${GREEN}================================================================${NC}"
echo ""
echo "Visit your repo:"
echo "  https://github.com/${GH_USER}/${GH_REPO}"
echo ""
echo "Recommended next steps:"
echo ""
echo "  1. Refresh the repo page — verify everything is there."
echo ""
echo "  2. Click the gear icon next to 'About' (top-right of repo page)."
echo ""
echo "  3. Set the description (paste this):"
echo ""
echo "     Model-based vs. model-free RL on highway-env: TD-MPC2 retains"
echo "     4x more training reward than SAC under cross-action-space"
echo "     zero-shot transfer. MLR 555 Group 13, AUS."
echo ""
echo "  4. Add topics (paste each, hit Enter between):"
echo "       reinforcement-learning, model-based-rl, world-models,"
echo "       autonomous-driving, tdmpc2, sac, highway-env, zero-shot-transfer,"
echo "       pytorch, gymnasium, mlr555, aus"
echo ""
echo "  5. Save changes."
echo ""
echo "  6. Submit the link!"
