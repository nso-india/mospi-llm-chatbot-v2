# Bot Auto Test

Automated testing tool for chatbot APIs. Creates a fresh session for each question to ensure cache is cleared between tests.

## Requirements

- Python 3.7+
- Required packages:
  ```bash
  pip install requests pandas openpyxl
  ```


## Usage

### Basic Usage
an excel/csv file containing questions under 'questions' column

```bash
python bot_auto_test.py questions.xlsx
```

### Input File Format

Excel (.xlsx) or CSV (.csv) file with a column named one of:
- `question`
- `questions`
- `q`

Example:
```
| question                          |
|-----------------------------------|
| What is the capital of France?    |
| How do I reset my password?       |
| What are your business hours?     |
```

### Command Line Options

```bash
# Custom bot URL
python bot_auto_test.py questions.xlsx

# Enable verbose output (shows each question/response)
python bot_auto_test.py questions.xlsx --verbose

```

### Output

Results are written to a JSONL file (default: `input_file.jsonl`). Each line contains:
```json
{
  "ts": "2025-01-23T10:30:45Z",
  "row_index": 0,
  "status": "ok",
  "phase": "ask",
  "question": "What is the capital of France?",
  "response": "The capital of France is Paris."
}
```

## Troubleshooting

### Question Column Not Found

**Problem**: `Could not find a question column`

**Solution**: 
- Rename your column to `question`, `questions`, or `q`
- Check for extra spaces in column names
- Verify file is not empty

### Empty Responses

**Problem**: Bot returns empty responses

**Solution**:
- Check JSONL file to see if errors are logged
- Verify bot API is working: test manually with curl
- Enable `--verbose` to see real-time progress
- Check bot logs for processing errors

### Script Hangs

**Problem**: Script appears stuck

**Solution**:
- Check JSONL file - it updates in real-time
- Bot may be processing a long query (default: waits indefinitely)
- Set `--read-timeout` to fail faster if needed
- Use `--verbose` to monitor progress
