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

`.claude/steering` - steering documents
`.claude/specs/{feature_name}` - feature spec
`.claude/specs/{feature_name}/tasks` - tasks

**Step 2: Set up an execution environment**

Set up a git worktree for a sub-agent with:

```sh
git branch feat/{feature_name}_{task_id}
git worktree worktrees/feat_{feature_name}_{task} feat/{feature_name}_{task_id}`
```

**Step 3: Execute with a sub-agent**

Execute one task at a time with a sub-agent in the worktree with the context:

```
## Steering Context
[PASTE THE STEERING CONTEXT HERE]

## Specification Context
[PASTE THE RELEVANT REQUIREMENTS AND DESIGN SECTIONS HERE]

## Task Details
[PASTE THE TASK DETAILS HERE]

## Notes

 - Follow all project conventions and leverage existing code
 - [CONTEXT FOR IMPLEMENTING IN THE WORKTREE]
```

**Step 4: Validate the implementation**

  - Run the required tests or benchmarks
  - Check for code violations in the git diff
  - If there are implementation issues re-do step 3 with an updated context

**Step 5: Mark the task as complete**

  - Present a completion summary and wait for approval
  - Merge and remove the git worktree
  - Mark the task as complete
  - Wait for approval before moving on to the next task
