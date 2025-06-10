# 🔒 Security Guidelines

## API Key Management

### ✅ DO
- Store API keys in `.env` files (already ignored by git)
- Use environment variables in code: `os.getenv("GEMINI_API_KEY")`
- Regularly rotate your API keys
- Use different keys for development and production

### ❌ DON'T
- **NEVER** commit API keys to git
- **NEVER** hardcode API keys in source code
- **NEVER** share API keys in chat, email, or documents
- **NEVER** put API keys in notebook outputs

## Quick Setup

1. **Get a new API key**: https://makersuite.google.com/app/apikey
2. **Update .env file**:
   ```bash
   GEMINI_API_KEY=your_actual_new_key_here
   GEMINI_MODEL=gemini-2.5-flash-preview-05-20
   ```

## If You Accidentally Leak a Key

1. **Immediately revoke the key** at Google AI Studio
2. **Generate a new key**
3. **Clean git history** (contact repo admin)
4. **Update .env with new key**

## Notebook Security

- Always use `os.getenv()` instead of hardcoded keys
- Clear outputs before committing notebooks
- Consider using `nbstripout` to auto-clean notebooks

## Prevention Tools

```bash
# Install git hooks to prevent accidental commits
pip install detect-secrets
detect-secrets scan --all-files
``` 