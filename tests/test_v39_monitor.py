import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    'v39_health_monitor', Path(__file__).resolve().parents[2] / 'scripts' / 'monitor_v39.py'
)
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


def test_failed_service_and_exited_container_are_visible():
    assert monitor.service_warnings('ActiveState=failed\nResult=exit-code', None, False)
    assert monitor.service_warnings('ActiveState=active\nResult=success',
                                    {'Status': 'exited', 'ExitCode': 137}, False)


def test_pause_evaluation_and_failed_probe_do_not_imply_training_failure():
    assert not monitor.service_warnings('ActiveState=inactive\nResult=success',
                                        {'Status': 'exited', 'ExitCode': 137}, False)
    assert not monitor.service_warnings('ActiveState=active\nResult=success',
                                        {'Status': 'exited', 'ExitCode': 0}, True)
    assert not monitor.service_warnings('ActiveState=active\nResult=success', None, False)
