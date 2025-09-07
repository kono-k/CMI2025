import yaml
import os

class CFG:
    """
    設定を管理するクラス。
    YAMLファイルから設定を読み込み、クラスの属性としてアクセス可能にする。
    """
    def __init__(self, yaml_path=None):
        if yaml_path:
            self.load_from_yaml(yaml_path)
        else:
            print("Warning: CFG initialized without a YAML path. Please load configuration manually or provide a path.")

    def load_from_yaml(self, yaml_path):
        """
        指定されたYAMLファイルから設定を読み込み、
        その内容をこのクラスの属性として設定する。
        ネストされたYAML構造も再帰的に処理する。
        """
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # YAMLの内容をクラスの属性として設定
        # 例: config_dict['data']['max_seq_len'] があれば、self.data.max_seq_len とアクセスできるようにする
        self._set_attributes_from_dict(config_dict, self)

    def _set_attributes_from_dict(self, data_dict, target_obj):
        """
        辞書の内容をターゲットオブジェクトの属性として再帰的に設定するヘルパー関数。
        """
        for key, value in data_dict.items():
            if isinstance(value, dict):
                # ネストされた辞書の場合、新しいConfigDictオブジェクトを作成し、再帰的に処理
                setattr(target_obj, key, ConfigDict())
                self._set_attributes_from_dict(value, getattr(target_obj, key))
            else:
                setattr(target_obj, key, value)

    def to_dict(self):
        """
        現在のCFGオブジェクトの内容を辞書形式で返す。
        WandBへのログ記録などに便利。
        """
        return self._to_dict_recursive(self)

    def _to_dict_recursive(self, obj):
        """
        オブジェクトの属性を辞書に変換する再帰ヘルパー関数。
        """
        if isinstance(obj, ConfigDict):
            return {k: self._to_dict_recursive(getattr(obj, k)) for k in obj.__dict__}
        else:
            return obj

    def __repr__(self):
        return f"CFG({self.to_dict()})"

    def __str__(self):
        return f"CFG({self.to_dict()})"


class ConfigDict:
    """
    ネストされたYAML構造をオブジェクトの属性としてアクセス可能にするためのダミークラス。
    例: cfg.data.max_seq_len のようにアクセスできるようにするため。
    """
    pass