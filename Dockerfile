# Stage 1: Builder
# This stage installs all Python dependencies, including any build-time OS packages.
FROM python:3.12-slim AS builder

# Install OS-level dependencies for packages that might have C extensions.
# Using --no-install-recommends helps keep the layer smaller.
# We clean apt cache and add flags to handle persistent GPG signature errors.
# We also clean up the archives after installation to save space.
RUN rm -rf /var/lib/apt/lists/* && \
    apt-get update -o Acquire::AllowInsecureRepositories=true && \
    apt-get install -y --allow-unauthenticated --no-install-recommends gcc build-essential && \
    rm -rf /var/cache/apt/archives/*

# Set the working directory
WORKDIR /app

# Create a virtual environment to isolate dependencies.
RUN python -m venv /opt/venv

# Add the venv to the PATH. This will be inherited by the runtime stage.
ENV PATH="/opt/venv/bin:$PATH"

# Copy only the requirements file to leverage Docker's layer caching.
COPY requirements.txt .

# Install Python dependencies into the virtual environment.
RUN pip install --no-cache-dir -r requirements.txt


# Stage 2: Runtime
# This stage creates the final, slim image with only the necessary files.
FROM python:3.12-slim

# Create a non-root user and group for better security.
RUN groupadd --system appgroup && useradd --system --gid appgroup appuser

# Set the working directory for the application.
WORKDIR /home/appuser/app

# Copy the virtual environment from the builder stage.
COPY --from=builder /opt/venv /opt/venv

# Copy the application source code into the container.
COPY . .

# Set environment variables.
# Ensures Python output is sent straight to the terminal.
ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/venv/bin:$PATH"

# Change ownership of the entire home directory to the non-root user.
RUN chown -R appuser:appgroup /home/appuser

# Switch to the non-root user.
USER appuser

# Define the entrypoint for the container.
ENTRYPOINT ["python", "run.py"]

# Set the default command, which can be easily overridden (e.g., with --help).
CMD ["--live"] 