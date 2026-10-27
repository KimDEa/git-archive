"""터미널에서 보기 좋은 로그 설정.
 
사용법:
    from log_config import logger
 
    logger.info("시작")
    try:
        ...
    except Exception:
        logger.exception("처리 중 오류")
"""
 
import logging
import sys
 
# 레벨별 색상 (터미널 ANSI 색 코드)
COLORS = {
    "DEBUG": "\033[36m",     # 청록
    "INFO": "\033[32m",      # 초록
    "WARNING": "\033[33m",   # 노랑
    "ERROR": "\033[31m",     # 빨강
    "CRITICAL": "\033[41m",  # 빨간 배경
}
RESET = "\033[0m"
DIM = "\033[2m"
 
 
class ColorFormatter(logging.Formatter):
    """레벨 이름에 색을 입히고, 시간·위치는 흐리게 표시한다."""
 
    def format(self, record):
        color = COLORS.get(record.levelname, "")
        time = self.formatTime(record, "%H:%M:%S")
        location = f"{record.filename}:{record.lineno}"
        line = (
            f"{DIM}{time}{RESET} "
            f"{color}{record.levelname:<8}{RESET} "
            f"{DIM}{location}{RESET} | {record.getMessage()}"
        )
        # logger.exception()을 쓰면 에러 내용(Traceback)을 아래에 붙인다
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line
 
 
def get_logger(name="app", level=logging.DEBUG):
    log = logging.getLogger(name)
    if not log.handlers:  # 여러 번 import해도 로그가 중복 출력되지 않게
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(ColorFormatter())
        log.addHandler(handler)
        log.setLevel(level)
        log.propagate = False
    return log
 
 
logger = get_logger()