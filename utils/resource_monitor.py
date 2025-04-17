import os
import time
import psutil
import torch
import threading

class ResourceMonitor:
    def __init__(self, config):
        """
        初始化资源监控器
        
        Args:
            config: 资源监控配置
        """
        self.config = config
        self.monitoring = False
        self.stats = {
            'cpu_percent': 0.0,
            'memory_percent': 0.0,
            'gpu_memory_used': 0.0,
            'gpu_memory_total': 0.0,
            'gpu_memory_percent': 0.0,
            'gpu_utilization': 0.0
        }
        
        # 检查是否可以使用GPU
        self.has_gpu = torch.cuda.is_available()
        if self.has_gpu:
            self.num_gpus = torch.cuda.device_count()
        else:
            self.num_gpus = 0
            
        # 启动监控线程
        self.monitor_thread = None
        self.start_monitoring()
        
    def start_monitoring(self):
        """启动资源监控"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_resources)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            
    def stop_monitoring(self):
        """停止资源监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
            
    def get_stats(self):
        """获取当前资源统计信息"""
        return self.stats
    
    def _monitor_resources(self):
        """资源监控线程"""
        while self.monitoring:
            # CPU和内存使用情况
            self.stats['cpu_percent'] = psutil.cpu_percent()
            self.stats['memory_percent'] = psutil.virtual_memory().percent
            
            # GPU使用情况
            if self.has_gpu:
                for i in range(self.num_gpus):
                    # 获取GPU内存信息
                    gpu_memory = torch.cuda.memory_stats(i)
                    allocated = gpu_memory.get('allocated_bytes.all.current', 0)
                    reserved = gpu_memory.get('reserved_bytes.all.current', 0)
                    total = torch.cuda.get_device_properties(i).total_memory
                    
                    self.stats['gpu_memory_used'] = allocated
                    self.stats['gpu_memory_total'] = total
                    self.stats['gpu_memory_percent'] = (allocated / total) * 100
                    
                    # 获取GPU利用率（需要安装pynvml或使用subprocess调用nvidia-smi）
                    try:
                        import pynvml
                        pynvml.nvmlInit()
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        self.stats['gpu_utilization'] = util.gpu
                    except (ImportError, Exception):
                        # 如果pynvml不可用，尝试使用subprocess调用nvidia-smi
                        try:
                            import subprocess
                            result = subprocess.check_output(
                                ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                                encoding='utf-8'
                            )
                            self.stats['gpu_utilization'] = float(result.strip())
                        except Exception:
                            self.stats['gpu_utilization'] = 0.0
            
            # 打印资源使用情况（如果配置为打印）
            if self.config.print_stats:
                print(f"Resource Usage - CPU: {self.stats['cpu_percent']:.1f}%, "
                      f"Memory: {self.stats['memory_percent']:.1f}%, "
                      f"GPU Memory: {self.stats['gpu_memory_percent']:.1f}%, "
                      f"GPU Util: {self.stats['gpu_utilization']:.1f}%")
            
            # 休眠一段时间
            time.sleep(self.config.monitor_interval)
    
    def get_resource_recommendation(self):
        """根据资源使用情况提供优化建议"""
        recommendations = []
        
        # GPU内存使用过高
        if self.stats['gpu_memory_percent'] > self.config.gpu_memory_high_threshold:
            recommendations.append({
                'type': 'gpu_memory',
                'action': 'reduce_batch_size',
                'current': self.stats['gpu_memory_percent'],
                'threshold': self.config.gpu_memory_high_threshold
            })
            
            recommendations.append({
                'type': 'gpu_memory',
                'action': 'reduce_samples',
                'current': self.stats['gpu_memory_percent'],
                'threshold': self.config.gpu_memory_high_threshold
            })
        
        # GPU利用率过低
        if self.has_gpu and self.stats['gpu_utilization'] < self.config.gpu_util_low_threshold:
            recommendations.append({
                'type': 'gpu_utilization',
                'action': 'increase_batch_size',
                'current': self.stats['gpu_utilization'],
                'threshold': self.config.gpu_util_low_threshold
            })
        
        # CPU使用率过高
        if self.stats['cpu_percent'] > self.config.cpu_high_threshold:
            recommendations.append({
                'type': 'cpu',
                'action': 'reduce_workers',
                'current': self.stats['cpu_percent'],
                'threshold': self.config.cpu_high_threshold
            })
            
        return recommendations

            
