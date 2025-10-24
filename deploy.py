#!/usr/bin/env python3
"""
Deployment automation script for Latency Spike Investigator.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Optional

from startup_checks import StartupValidator


class DeploymentManager:
    """Manages deployment tasks and validation."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.validator = StartupValidator()
    
    def validate_deployment_readiness(self) -> bool:
        """Validate that the application is ready for deployment."""
        print("🔍 Validating deployment readiness...")
        
        success, results = self.validator.run_all_checks()
        
        if not success:
            print("❌ Deployment validation failed!")
            for error in results['errors']:
                print(f"  ERROR: {error}")
            return False
        
        if results['warnings']:
            print("⚠️  Deployment warnings:")
            for warning in results['warnings']:
                print(f"  WARNING: {warning}")
        
        print("✅ Deployment validation passed!")
        return True
    
    def prepare_render_deployment(self) -> bool:
        """Prepare files for Render deployment."""
        print("📦 Preparing Render deployment...")
        
        # Check if render.yaml exists
        render_config = self.project_root / "render.yaml"
        if not render_config.exists():
            print("❌ render.yaml not found!")
            return False
        
        # Validate requirements.txt
        requirements_file = self.project_root / "requirements.txt"
        if not requirements_file.exists():
            print("❌ requirements.txt not found!")
            return False
        
        print("✅ Render deployment files ready!")
        print(f"📋 Next steps:")
        print(f"  1. Push code to your Git repository")
        print(f"  2. Connect repository to Render")
        print(f"  3. Configure environment variables in Render dashboard")
        print(f"  4. Deploy using the render.yaml configuration")
        
        return True
    
    def prepare_huggingface_deployment(self) -> bool:
        """Prepare files for Hugging Face Spaces deployment."""
        print("🤗 Preparing Hugging Face Spaces deployment...")
        
        # Check if README_HF.md exists
        hf_readme = self.project_root / "README_HF.md"
        if not hf_readme.exists():
            print("❌ README_HF.md not found!")
            return False
        
        # Check main app file
        app_file = self.project_root / "app.py"
        if not app_file.exists():
            print("❌ app.py not found!")
            return False
        
        print("✅ Hugging Face Spaces deployment files ready!")
        print(f"📋 Next steps:")
        print(f"  1. Create a new Space on Hugging Face")
        print(f"  2. Upload all files to the Space repository")
        print(f"  3. Rename README_HF.md to README.md in the Space")
        print(f"  4. Configure secrets in Space settings")
        print(f"  5. The Space will automatically build and deploy")
        
        return True
    
    def prepare_docker_deployment(self) -> bool:
        """Prepare and test Docker deployment."""
        print("🐳 Preparing Docker deployment...")
        
        # Check if Dockerfile exists
        dockerfile = self.project_root / "Dockerfile"
        if not dockerfile.exists():
            print("❌ Dockerfile not found!")
            return False
        
        # Test Docker build (if Docker is available)
        try:
            result = subprocess.run(
                ["docker", "--version"], 
                capture_output=True, 
                text=True, 
                check=True
            )
            print(f"✅ Docker available: {result.stdout.strip()}")
            
            # Optionally build the image
            build_image = input("Build Docker image now? (y/N): ").lower().strip()
            if build_image == 'y':
                print("🔨 Building Docker image...")
                build_result = subprocess.run(
                    ["docker", "build", "-t", "latency-spike-investigator", "."],
                    cwd=self.project_root
                )
                if build_result.returncode == 0:
                    print("✅ Docker image built successfully!")
                else:
                    print("❌ Docker build failed!")
                    return False
                    
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️  Docker not available - skipping build test")
        
        print("✅ Docker deployment files ready!")
        print(f"📋 Next steps:")
        print(f"  1. Build image: docker build -t latency-spike-investigator .")
        print(f"  2. Run container: docker run -p 8501:8501 latency-spike-investigator")
        print(f"  3. Or use docker-compose for full stack deployment")
        
        return True
    
    def create_env_template(self) -> bool:
        """Create environment variable template."""
        print("📝 Creating environment template...")
        
        env_template = self.project_root / ".env.template"
        template_content = """# Latency Spike Investigator Environment Configuration

# Required for AI features (optional - app runs in demo mode without it)
GEMINI_API_KEY=your_gemini_api_key_here

# Optional APM integrations
NEW_RELIC_API_KEY=your_newrelic_api_key_here
DATADOG_API_KEY=your_datadog_api_key_here
DATADOG_APP_KEY=your_datadog_app_key_here

# Application Configuration
SPIKE_THRESHOLD_MS=1000
CORRELATION_WINDOW_MINUTES=15
CACHE_TTL_SECONDS=300
GEMINI_REQUESTS_PER_MINUTE=60

# Database Configuration
SQLITE_DB_PATH=data/latency_investigator.db
REDIS_URL=redis://localhost:6379

# Deployment Configuration
PORT=8501
HOST=0.0.0.0
DEBUG=false
ENVIRONMENT=development
"""
        
        with open(env_template, 'w') as f:
            f.write(template_content)
        
        print(f"✅ Environment template created: {env_template}")
        print(f"📋 Copy to .env and configure your values")
        
        return True
    
    def run_deployment_checklist(self, platform: str) -> bool:
        """Run platform-specific deployment checklist."""
        print(f"🚀 Running {platform} deployment checklist...")
        
        # Common validation
        if not self.validate_deployment_readiness():
            return False
        
        # Platform-specific preparation
        if platform == "render":
            return self.prepare_render_deployment()
        elif platform == "huggingface":
            return self.prepare_huggingface_deployment()
        elif platform == "docker":
            return self.prepare_docker_deployment()
        else:
            print(f"❌ Unknown platform: {platform}")
            return False


def main():
    """Main deployment script entry point."""
    parser = argparse.ArgumentParser(description="Deployment automation for Latency Spike Investigator")
    parser.add_argument(
        "platform", 
        choices=["render", "huggingface", "docker", "local"],
        help="Deployment platform"
    )
    parser.add_argument(
        "--validate-only", 
        action="store_true",
        help="Only run validation checks"
    )
    parser.add_argument(
        "--create-env", 
        action="store_true",
        help="Create environment template file"
    )
    
    args = parser.parse_args()
    
    manager = DeploymentManager()
    
    # Create environment template if requested
    if args.create_env:
        manager.create_env_template()
        return
    
    # Run validation only if requested
    if args.validate_only:
        success = manager.validate_deployment_readiness()
        sys.exit(0 if success else 1)
    
    # Run full deployment checklist
    if args.platform == "local":
        # Local development setup
        print("🏠 Setting up local development environment...")
        success = manager.validate_deployment_readiness()
        if success:
            print("✅ Local environment ready!")
            print("📋 Start the application with: streamlit run app.py")
    else:
        success = manager.run_deployment_checklist(args.platform)
    
    if success:
        print(f"\n🎉 {args.platform.title()} deployment preparation completed!")
    else:
        print(f"\n❌ {args.platform.title()} deployment preparation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()