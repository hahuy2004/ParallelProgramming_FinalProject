# Python wrapper for CIFAR-10 Autoencoder + SVM project
# Allows running C++/CUDA code from Python/Jupyter

import subprocess
import os
import sys
import time
from pathlib import Path

class CIFARAutoencoderPipeline:
    """Python wrapper for running the CIFAR-10 Autoencoder pipeline"""
    
    def __init__(self, project_dir="/home/hahuy2004/LT_song_song/LT/Project"):
        self.project_dir = Path(project_dir)
        self.build_dir = self.project_dir / "build"
        self.weights_dir = self.project_dir / "weights"
        self.results_dir = self.project_dir / "results"
        
        # Ensure directories exist
        self.weights_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
    
    def check_setup(self):
        """Check if environment is properly set up"""
        print("=== Checking Setup ===")
        
        # Check CIFAR-10 dataset
        cifar_dir = self.project_dir / "cifar-10-batches-bin"
        if not cifar_dir.exists():
            print("❌ CIFAR-10 dataset not found!")
            return False
        print(f"✅ CIFAR-10 dataset found: {cifar_dir}")
        
        # Check build directory
        if not self.build_dir.exists():
            print("❌ Build directory not found! Run 'make' first.")
            return False
        print(f"✅ Build directory found: {self.build_dir}")
        
        # Check CUDA
        result = subprocess.run(["nvcc", "--version"], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ CUDA found: {result.stdout.split('release')[1].split(',')[0].strip()}")
        else:
            print("⚠️  CUDA not found - GPU phases will not work")
        
        # Check libsvm
        libsvm_dir = self.project_dir / "third_party" / "libsvm"
        if not libsvm_dir.exists():
            print("⚠️  LIBSVM not found - Phase 4 will not work")
        else:
            print(f"✅ LIBSVM found: {libsvm_dir}")
        
        return True
    
    def compile_all(self):
        """Compile all phases"""
        print("\n=== Compiling All Phases ===")
        result = subprocess.run(["make", "all"], 
                               cwd=self.project_dir,
                               capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Compilation successful!")
            print(result.stdout)
            return True
        else:
            print("❌ Compilation failed!")
            print(result.stderr)
            return False
    
    def compile_phase(self, phase):
        """Compile a specific phase"""
        print(f"\n=== Compiling Phase {phase} ===")
        target = f"phase{phase}"
        result = subprocess.run(["make", target], 
                               cwd=self.project_dir,
                               capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Phase {phase} compiled successfully!")
            return True
        else:
            print(f"❌ Phase {phase} compilation failed!")
            print(result.stderr)
            return False
    
    def run_phase1_cpu(self):
        """Run Phase 1: CPU Baseline"""
        print("\n" + "="*50)
        print("PHASE 1: CPU Baseline")
        print("="*50)
        
        executable = self.build_dir / "phase1"
        if not executable.exists():
            print("❌ phase1 executable not found! Compile it first.")
            return None
        
        start = time.time()
        result = subprocess.run([str(executable)], 
                               cwd=self.project_dir,
                               capture_output=True, text=True)
        elapsed = time.time() - start
        
        print(result.stdout)
        if result.returncode != 0:
            print("❌ Phase 1 failed!")
            print(result.stderr)
            return None
        
        return {"phase": 1, "time": elapsed, "output": result.stdout}
    
    def run_phase2_gpu(self):
        """Run Phase 2: Naive GPU Implementation"""
        print("\n" + "="*50)
        print("PHASE 2: Naive GPU Implementation")
        print("="*50)
        
        executable = self.build_dir / "phase2"
        if not executable.exists():
            print("❌ phase2 executable not found! Compile it first.")
            return None
        
        start = time.time()
        result = subprocess.run([str(executable)], 
                               cwd=self.project_dir,
                               capture_output=True, text=True)
        elapsed = time.time() - start
        
        print(result.stdout)
        if result.returncode != 0:
            print("❌ Phase 2 failed!")
            print(result.stderr)
            return None
        
        return {"phase": 2, "time": elapsed, "output": result.stdout}
    
    def run_phase3_optimized(self):
        """Run Phase 3: Optimized GPU Implementation"""
        print("\n" + "="*50)
        print("PHASE 3: Optimized GPU Implementation")
        print("="*50)
        
        executable = self.build_dir / "phase3"
        if not executable.exists():
            print("❌ phase3 executable not found! Compile it first.")
            return None
        
        start = time.time()
        result = subprocess.run([str(executable)], 
                               cwd=self.project_dir,
                               capture_output=True, text=True)
        elapsed = time.time() - start
        
        print(result.stdout)
        if result.returncode != 0:
            print("❌ Phase 3 failed!")
            print(result.stderr)
            return None
        
        return {"phase": 3, "time": elapsed, "output": result.stdout}
    
    def run_phase4_svm(self, use_optimized=True):
        """Run Phase 4: SVM Classification"""
        print("\n" + "="*50)
        print("PHASE 4: SVM Classification")
        print("="*50)
        
        executable = self.build_dir / "phase4"
        if not executable.exists():
            print("❌ phase4 executable not found! Compile it first.")
            return None
        
        # Choose weights file
        if use_optimized:
            weights_file = "weights/autoencoder_gpu_optimized.weights"
        else:
            weights_file = "weights/autoencoder_gpu.weights"
        
        print(f"Using weights: {weights_file}")
        
        start = time.time()
        result = subprocess.run([str(executable), weights_file], 
                               cwd=self.project_dir,
                               capture_output=True, text=True)
        elapsed = time.time() - start
        
        print(result.stdout)
        if result.returncode != 0:
            print("❌ Phase 4 failed!")
            print(result.stderr)
            return None
        
        return {"phase": 4, "time": elapsed, "output": result.stdout}
    
    def run_full_pipeline(self, skip_cpu=False):
        """Run all phases in sequence"""
        print("\n" + "="*60)
        print("RUNNING FULL PIPELINE")
        print("="*60)
        
        results = {}
        
        # Phase 1: CPU (optional)
        if not skip_cpu:
            result = self.run_phase1_cpu()
            if result:
                results["phase1"] = result
        
        # Phase 2: Naive GPU
        result = self.run_phase2_gpu()
        if result:
            results["phase2"] = result
        else:
            print("⚠️  Skipping remaining phases due to Phase 2 failure")
            return results
        
        # Phase 3: Optimized GPU
        result = self.run_phase3_optimized()
        if result:
            results["phase3"] = result
        
        # Phase 4: SVM Classification
        result = self.run_phase4_svm(use_optimized=True)
        if result:
            results["phase4"] = result
        
        # Print summary
        print("\n" + "="*60)
        print("PIPELINE SUMMARY")
        print("="*60)
        for phase, data in results.items():
            print(f"{phase.upper()}: {data['time']:.2f}s")
        
        return results
    
    def profile_with_nsys(self, phase=3):
        """Profile a phase with NVIDIA Nsight Systems"""
        print(f"\n=== Profiling Phase {phase} with nsys ===")
        
        executable = self.build_dir / f"phase{phase}"
        if not executable.exists():
            print(f"❌ phase{phase} executable not found!")
            return False
        
        output_file = self.results_dir / f"phase{phase}_profile"
        
        cmd = [
            "nsys", "profile",
            "--output", str(output_file),
            "--force-overwrite", "true",
            str(executable)
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=self.project_dir)
        
        if result.returncode == 0:
            print(f"✅ Profiling completed! Report saved to {output_file}.nsys-rep")
            print(f"Open with: nsys-ui {output_file}.nsys-rep")
            return True
        else:
            print("❌ Profiling failed!")
            return False
    
    def clean(self):
        """Clean build artifacts"""
        print("\n=== Cleaning Build ===")
        result = subprocess.run(["make", "clean"], 
                               cwd=self.project_dir,
                               capture_output=True, text=True)
        print(result.stdout)
        return result.returncode == 0


# Convenience functions for Jupyter
def quick_start():
    """Quick start: compile and run all phases"""
    pipeline = CIFARAutoencoderPipeline()
    
    if not pipeline.check_setup():
        print("\n❌ Setup check failed! Fix issues above before continuing.")
        return None
    
    if not pipeline.compile_all():
        print("\n❌ Compilation failed! Check errors above.")
        return None
    
    return pipeline.run_full_pipeline(skip_cpu=True)


def run_phase(phase_num):
    """Run a specific phase"""
    pipeline = CIFARAutoencoderPipeline()
    
    methods = {
        1: pipeline.run_phase1_cpu,
        2: pipeline.run_phase2_gpu,
        3: pipeline.run_phase3_optimized,
        4: pipeline.run_phase4_svm,
    }
    
    if phase_num not in methods:
        print(f"❌ Invalid phase number: {phase_num}")
        return None
    
    # Compile first
    if not pipeline.compile_phase(phase_num):
        return None
    
    return methods[phase_num]()


if __name__ == "__main__":
    # Command-line interface
    if len(sys.argv) < 2:
        print("Usage: python run_pipeline.py [command]")
        print("Commands:")
        print("  check      - Check setup")
        print("  compile    - Compile all phases")
        print("  phase1     - Run Phase 1 (CPU)")
        print("  phase2     - Run Phase 2 (Naive GPU)")
        print("  phase3     - Run Phase 3 (Optimized GPU)")
        print("  phase4     - Run Phase 4 (SVM)")
        print("  all        - Run full pipeline")
        print("  profile    - Profile Phase 3")
        print("  clean      - Clean build")
        sys.exit(1)
    
    pipeline = CIFARAutoencoderPipeline()
    command = sys.argv[1].lower()
    
    if command == "check":
        pipeline.check_setup()
    elif command == "compile":
        pipeline.compile_all()
    elif command == "phase1":
        pipeline.run_phase1_cpu()
    elif command == "phase2":
        pipeline.run_phase2_gpu()
    elif command == "phase3":
        pipeline.run_phase3_optimized()
    elif command == "phase4":
        pipeline.run_phase4_svm()
    elif command == "all":
        pipeline.run_full_pipeline()
    elif command == "profile":
        pipeline.profile_with_nsys(3)
    elif command == "clean":
        pipeline.clean()
    else:
        print(f"❌ Unknown command: {command}")
        sys.exit(1)
