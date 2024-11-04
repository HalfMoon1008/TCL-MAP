import importlib
from easydict import EasyDict


class ParamManager:
    """
    파라미터 관리를 위한 클래스
    인스턴스 생성 시 전달받은 인자를 EasyDict로 변환하여 self.args에 저장
    """
    
    def __init__(self, args):
        # 인자로 받은 args를 EasyDict로 변환하여 self.args에 저장
        # EasyDict는 일반 딕셔너리처럼 사용할 수 있지만, 점 표기법으로도 접근할 수 있게 해줌
        self.args = EasyDict(dict(vars(args)))   


def add_config_param(old_args, config_file_name=None):
    """
    기존 파라미터(old_args)에 추가 구성 파일에서 불러온 파라미터를 병합하여 새로운 파라미터를 반환하는 함수
    
    Args:
        old_args: 기존 파라미터로, 기본 인자를 포함한 EasyDict 형식의 파라미터 객체
        config_file_name: 불러올 구성 파일의 이름 (기본값: None)
        
    Returns:
        기존 파라미터와 구성 파일에서 불러온 파라미터를 병합한 새로운 EasyDict 객체를 반환
    """
    
    # config_file_name이 지정되지 않았다면 old_args에서 config_file_name을 가져옴
    if config_file_name is None:
        config_file_name = old_args.config_file_name
        
    # config_file_name이 '.py'로 끝나는 경우 확장자를 제거하고 모듈 이름을 생성
    if config_file_name.endswith('.py'):
        module_name = '.' + config_file_name[:-3]
    else:
        module_name = '.' + config_file_name

    # importlib을 사용하여 'configs' 패키지에서 해당 모듈을 동적으로 불러옴
    # 'configs'라는 패키지 아래에 설정 파일들이 위치하고 있는 것으로 가정
    config = importlib.import_module(module_name, 'configs')

    # 불러온 설정 모듈에서 'Param' 클래스를 가져옵니다.
    config_param = config.Param
    
    # 'Param' 클래스의 인스턴스를 생성하고, old_args를 인자로 전달하여 구성 파일의 파라미터를 불러옴
    method_args = config_param(old_args)
    
    # 기존 파라미터(old_args)와 불러온 파라미터(method_args.hyper_param)를 병합하여 새로운 EasyDict 객체를 생성
    # 병합 시 old_args의 파라미터에 method_args의 hyper_param 속성을 덮어씀
    new_args = EasyDict(dict(old_args, **method_args.hyper_param))
    
    return new_args
