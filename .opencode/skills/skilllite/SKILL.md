---
name: skilllite
description: Execute code or pre-defined skills in a secure sandbox. Use when running untrusted code, network requests, or data processing.
---

## Overview

SkillLite provides a secure sandbox execution environment. Code runs isolated in a system-level sandbox (macOS Seatbelt / Linux Namespace), preventing malicious code from affecting the host.

## When to use SkillLite vs bash

| Scenario | bash | SkillLite |
|----------|------|-----------|
| git operations | ✅ | |
| Read project files | ✅ | |
| Execute user-provided code | | ✅ |
| Network/API calls | | ✅ |
| Data analysis | | ✅ |
| Run untrusted scripts | | ✅ |
| Execute potentially dangerous commands | | ✅ |

## Available Tools

### 1. skilllite_execute_code
Execute arbitrary code (Python/JavaScript/Bash) in sandbox.

### 2. skilllite_run_skill
Execute a pre-defined skill.

### 3. skilllite_list_skills
List all available pre-defined skills. No params.

### 4. skilllite_get_skill_info
Get detailed info for a skill, including input schema.

### 5. skilllite_scan_code
Scan code safety only, no execution.

## Pre-defined Skills

- (No pre-defined skills found. Use skilllite_execute_code for code execution.)

