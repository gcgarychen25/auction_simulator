# 🚀 Quick Setup Instructions

## Step 1: Set Up Environment Variables

1. **Rename the template file:**
   ```bash
   mv env_template .env
   ```

2. **Edit the `.env` file:**
   - Open `.env` in Cursor IDE
   - Replace `your_api_key_here` with your actual Gemini API key
   - Get your API key from: https://makersuite.google.com/app/apikey
   - (Optional) Uncomment and set `GEMINI_MODEL` to use a specific model

3. **Your `.env` file should look like:**
   ```
   GEMINI_API_KEY=AIzaSy...your_actual_key_here
   
   # Optional: Specify your preferred model
   GEMINI_MODEL=gemini-2.5-flash-preview-05-20
   
   LOG_LEVEL=INFO
   DEFAULT_EPISODES=1
   DEFAULT_VERBOSE=true
   ```

## Step 2: Install Dependencies

```bash
# Make sure python-dotenv is installed
pip install python-dotenv

# Install all requirements
pip install -r requirements.txt
```

## Step 3: Using the Notebook in Cursor IDE

1. **Open notebook.ipynb in Cursor IDE**
2. **Select your Python interpreter** (make sure it's the auction_simulator conda env)
3. **Run the first cell** to load environment variables
4. **You should see:** `✅ Environment variables loaded from .env file`
5. **Continue running cells** to test the simulation

## Step 4: Verify Setup

Run this in terminal to test:
```bash
python run.py --episodes 1 --verbose
```

## 🎯 You're Ready!

- ✅ Environment variables loaded from `.env`
- ✅ API key configured (optional but recommended)
- ✅ Gemini model configured (defaults to gemini-1.5-flash)
- ✅ Dependencies installed  
- ✅ Notebook ready to use in Cursor IDE

## 🤖 Model Configuration

The system now supports multiple Gemini models:
- **gemini-1.5-flash** (default, fast and efficient)
- **gemini-1.5-pro** (more capable, slower)
- **gemini-2.0-flash-exp** (experimental)
- **gemini-2.5-flash-preview-05-20** (preview model)

To use a specific model, uncomment and set `GEMINI_MODEL` in your `.env` file.

## 🔧 Troubleshooting

**If you see "API key not found":**
- Check that `.env` file exists
- Verify the API key format in `.env`
- Re-run the first notebook cell

**If you see "model not found" errors:**
- Check that your model name is correct
- Try using the default `gemini-1.5-flash`
- Verify your API key has access to the model

**If imports fail:**
- Make sure you're in the right conda environment
- Run `pip install -r requirements.txt` again

**If notebook doesn't work in Cursor:**
- Make sure Cursor has the Python extension installed
- Select the correct Python interpreter (bottom-left in Cursor)
- Restart the kernel if needed 