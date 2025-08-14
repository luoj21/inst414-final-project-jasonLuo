import logging

logging.basicConfig(       
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    filename='logs/LOGS.log',
    level=logging.DEBUG
)
my_logger = logging.getLogger(__name__)