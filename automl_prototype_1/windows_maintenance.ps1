$ErrorActionPreference = "Stop"

Write-Host "=========================================================" -ForegroundColor Cyan
Write-Host "      WINDOWS ENVIRONMENT MAINTENANCE & AUDIT" -ForegroundColor Cyan
Write-Host "=========================================================" -ForegroundColor Cyan

# 1. JAVA / MAVEN DEPENDENCIES
Write-Host "`n[1/4] Checking Java Project Dependencies..." -ForegroundColor Yellow
if (Get-Command mvn -ErrorAction SilentlyContinue) {
    Write-Host "      Running Maven Dependency Audit (this may take a minute)..."
    # Run versions:display-dependency-updates to find newer versions
    cmd /c mvn versions:display-dependency-updates -DallowSnapshots=false -f pom.xml
    
    # Clean Install to refresh local artifacts
    Write-Host "`n      Refeshing Local Artifacts (Clean Install)..."
    cmd /c mvn clean install -DskipTests
} else {
    Write-Host "      Maven (mvn) not found in PATH. Skipping." -ForegroundColor Red
}

# 2. PYTHON (WINDOWS NATIVE)
Write-Host "`n[2/4] Checking Windows Python Environment..." -ForegroundColor Yellow
if (Get-Command python -ErrorAction SilentlyContinue) {
    $pyVer = python --version
    Write-Host "      Found: $pyVer"
    
    # Upgrade PIP
    Write-Host "      Upgrading pip..."
    python -m pip install --upgrade pip

    # Upgrade core libraries if possible (pandas, numpy, etc. if installed globally)
    # detecting modules
    $hasPandas = python -c "import pkgutil; print(1 if pkgutil.find_loader('pandas') else 0)"
    if ($hasPandas -eq "1") {
        Write-Host "      Upgrading global pandas/numpy..."
        python -m pip install --upgrade pandas numpy scikit-learn
    }
} else {
    Write-Host "      Python not found in Windows PATH." -ForegroundColor Gray
}

# 3. GPU DRIVERS
Write-Host "`n[3/4] Checking GPU Driver Status..." -ForegroundColor Yellow
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    nvidia-smi --query-gpu=driver_version,gpu_name --format=csv,noheader
} else {
    Write-Host "      NVIDIA Driver tools not found." -ForegroundColor Red
}

# 4. LANGUAGE LEVEL CHECK
Write-Host "`n[4/4] Auditing Language Level..." -ForegroundColor Yellow
# Simple regex check on pom.xml for maven.compiler.release
$pomContent = Get-Content pom.xml -Raw
if ($pomContent -match "<maven.compiler.release>(\d+)</maven.compiler.release>") {
    $currentJava = $matches[1]
    Write-Host "      Current Project Java Language Level: JDK $currentJava"
    
    if ([int]$currentJava -lt 21) {
        Write-Host "      NOTICE: Java 21 LTS is available. You are using $currentJava." -ForegroundColor Magenta
        Write-Host "      Upgrade is recommended if you have JDK 21 installed."
    } else {
        Write-Host "      You are using a modern Java version." -ForegroundColor Green
    }
}

Write-Host "`n=========================================================" -ForegroundColor Cyan
Write-Host "      MAINTENANCE COMPLETE" -ForegroundColor Cyan
Write-Host "=========================================================" -ForegroundColor Cyan
Write-Host "Review the Maven output above for available library upgrades."
