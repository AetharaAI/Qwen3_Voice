"""
GPU monitoring utilities using nvidia-ml-py.
"""

import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Try to import pynvml
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    logger.warning("pynvml not available, GPU monitoring disabled")


@dataclass
class GPUStats:
    """GPU statistics data class."""
    device_id: int
    name: str
    total_memory_mb: int
    used_memory_mb: int
    free_memory_mb: int
    utilization_percent: float
    temperature_celsius: Optional[int] = None
    power_draw_watts: Optional[float] = None


class GPUMonitor:
    """GPU monitoring utility class."""
    
    _instance: Optional['GPUMonitor'] = None
    _initialized: bool = False
    
    def __new__(cls) -> 'GPUMonitor':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if GPUMonitor._initialized:
            return
        
        self._available = False
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self._available = True
                self._device_count = pynvml.nvmlDeviceGetCount()
                logger.info(f"NVML initialized, found {self._device_count} GPU(s)")
            except Exception as e:
                logger.error(f"Failed to initialize NVML: {e}")
        
        GPUMonitor._initialized = True
    
    @property
    def available(self) -> bool:
        """Check if GPU monitoring is available."""
        return self._available
    
    def get_device_count(self) -> int:
        """Get the number of GPU devices."""
        if not self._available:
            return 0
        return self._device_count
    
    def get_gpu_stats(self, device_id: int) -> Optional[GPUStats]:
        """Get statistics for a specific GPU device."""
        if not self._available:
            return None
        
        try:
            if device_id >= self._device_count:
                logger.error(f"Invalid device ID {device_id}, max is {self._device_count - 1}")
                return None
            
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            
            # Get device name
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode('utf-8')
            
            # Get memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_mb = mem_info.total // (1024 * 1024)
            used_mb = mem_info.used // (1024 * 1024)
            free_mb = mem_info.free // (1024 * 1024)
            
            # Get utilization
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            util_percent = utilization.gpu
            
            # Get temperature (optional)
            temperature = None
            try:
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                pass
            
            # Get power draw (optional)
            power_draw = None
            try:
                power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert mW to W
            except Exception:
                pass
            
            return GPUStats(
                device_id=device_id,
                name=name,
                total_memory_mb=total_mb,
                used_memory_mb=used_mb,
                free_memory_mb=free_mb,
                utilization_percent=util_percent,
                temperature_celsius=temperature,
                power_draw_watts=power_draw
            )
        
        except Exception as e:
            logger.error(f"Failed to get GPU stats for device {device_id}: {e}")
            return None
    
    def get_all_gpu_stats(self) -> Dict[int, GPUStats]:
        """Get statistics for all GPU devices."""
        stats = {}
        if not self._available:
            return stats
        
        for i in range(self._device_count):
            gpu_stats = self.get_gpu_stats(i)
            if gpu_stats:
                stats[i] = gpu_stats
        
        return stats
    
    def to_dict(self, device_id: int) -> Dict[str, Any]:
        """Convert GPU stats to dictionary."""
        stats = self.get_gpu_stats(device_id)
        if stats is None:
            return {
                "gpu_id": device_id,
                "available": False,
                "vram_total_mb": 0,
                "vram_used_mb": 0,
                "vram_free_mb": 0,
                "utilization_percent": 0.0
            }
        
        result = {
            "gpu_id": stats.device_id,
            "available": True,
            "name": stats.name,
            "vram_total_mb": stats.total_memory_mb,
            "vram_used_mb": stats.used_memory_mb,
            "vram_free_mb": stats.free_memory_mb,
            "utilization_percent": stats.utilization_percent
        }
        
        if stats.temperature_celsius is not None:
            result["temperature_celsius"] = stats.temperature_celsius
        if stats.power_draw_watts is not None:
            result["power_draw_watts"] = stats.power_draw_watts
        
        return result
    
    def shutdown(self):
        """Shutdown NVML."""
        if self._available and PYNVML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
                logger.info("NVML shutdown complete")
            except Exception as e:
                logger.error(f"Error shutting down NVML: {e}")


# Global instance
gpu_monitor = GPUMonitor()