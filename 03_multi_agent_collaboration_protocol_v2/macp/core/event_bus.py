"""
事件总线
Agent间消息传递和事件通知
"""
import threading
from typing import Dict, List, Callable, Any, Optional
from collections import defaultdict


class EventBus:
    """
    事件总线
    
    提供发布-订阅模式的消息传递
    """
    
    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._lock = threading.RLock()
        self._history: List[Dict] = []
        self._max_history = 1000
    
    def subscribe(self, event_type: str, handler: Callable[[Dict], None]):
        """
        订阅事件
        
        Args:
            event_type: 事件类型
            handler: 事件处理器
        """
        with self._lock:
            self._subscribers[event_type].append(handler)
    
    def unsubscribe(self, event_type: str, handler: Callable[[Dict], None]):
        """取消订阅"""
        with self._lock:
            if handler in self._subscribers[event_type]:
                self._subscribers[event_type].remove(handler)
    
    def publish(self, event_type: str, data: Dict[str, Any]):
        """
        发布事件
        
        Args:
            event_type: 事件类型
            data: 事件数据
        """
        event = {
            "type": event_type,
            "data": data,
        }
        
        # 保存历史
        with self._lock:
            self._history.append(event)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]
        
        # 通知订阅者
        handlers = self._subscribers.get(event_type, [])
        for handler in handlers:
            try:
                handler(data)
            except Exception as e:
                print(f"Event handler error: {e}")
    
    def get_history(self, event_type: Optional[str] = None, n: int = 100) -> List[Dict]:
        """获取事件历史"""
        with self._lock:
            history = self._history
            if event_type:
                history = [e for e in history if e["type"] == event_type]
            return history[-n:]
    
    def clear_history(self):
        """清空历史"""
        with self._lock:
            self._history.clear()
