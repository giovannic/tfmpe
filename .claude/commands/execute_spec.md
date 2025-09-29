# Execute Spec Command

Execute specific tasks from the approved task list.

## Usage
```
/execute_spec [feature_name] [task_id] 
```

## Phase Overview
**Your Role**: Execute tasks systematically with validation

This is Phase 4 of the spec workflow. Your goal is to implement individual tasks from the approved task list, one at a time.

## Instructions

**Execution Steps**:

**Step 1: Load Context**

`.claude/steering/*` - steering documents
`.claude/specs/{feature_name}/*` - feature specs
`.claude/specs/{feature_name}/tasks.md` - tasks

**Step 2: Set up an execution environment**

Set up a git worktree with:

```sh
git branch feat/{feature_name}_{task_id}
git worktree worktrees/feat_{feature_name}_{task} feat/{feature_name}_{task_id}`
```
**Step 3: Execute task**

  - Execute one task at a time in the worktree.
  - Make sure to prepend `. ../../env/bin/activate` to your python calls

**Step 4: Validate the implementation**

  - Run the required tests and/or benchmarks
  - Check for code violations
    - Duplicated code
    - Unused declarations
    - Imports at the top of the file

**Step 5: Mark the task as complete**

  - Present a completion summary and wait for approval
  - Mark the task as complete in `.claude/specs/{feature_name}/tasks.md`
  - Merge and remove the git worktree
