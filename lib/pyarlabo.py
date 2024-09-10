"""
pyarlabo
================================

This library manipulates and analyzes time-series coordinate files (CSV format)
output from the AR-LABO system.

Author: Okkn
Date: 2024-02-24
Version: 1.0.0
"""

from typing import List, Dict, Tuple, Optional, Union, Iterator
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# グローバル変数を汚さないようにするコンテキストを定義する
class TempVarContext:
    def __init__(self, **variables):
        super().__setattr__('_variables', variables)

    def __enter__(self):
        return self  # コンテキスト自身を返す

    def __setattr__(self, name, value):
        self._variables[name] = value

    def __getattr__(self, name):
        if name == '_variables':
            try:
                return super().__getattribute__('_variables')
            except AttributeError:
                raise AttributeError("This context is no longer available")
        try:
            return self._variables[name]
        except KeyError:
            raise AttributeError(f"`{name}` is not defined in this context")

    def __exit__(self, exc_type, exc_val, traceback):
        del self._variables
        return False


class Project:
    """実験全体のマウスのリストや、実験の条件を管理するクラス"""
    def __init__(self, mice_list: pd.DataFrame,
                 comp_col: str, comparison: Optional[str] = None,
                 marker_id_col: str = "marker_id",
                 session_col: str = "session", camera_col: str = "camera"):
        self.mice_list: pd.DataFrame = mice_list.copy()

        # comp_colがself.mice_listにない場合はエラーを出す
        if comp_col not in self.mice_list:
            raise ValueError(f"`{comp_col}` is not in the mice list.")
        self.comp_col: str = comp_col
        self.comparison: Optional[str] = comparison

        if marker_id_col not in self.mice_list:
            raise ValueError(f"`{marker_id_col}` is not in the mice list.")
        self.marker_id_col: str = marker_id_col
        self.mice_list[self.marker_id_col] = \
            self.mice_list[self.marker_id_col].map(str)  # marker_idを文字列に

        # session_colがself.mice_listにない場合はエラーを出す
        if session_col not in self.mice_list:
            raise ValueError(f"`{session_col}` is not in the mice list.")
        self.session_col: str = session_col
        self.mice_list[self.session_col] = self.mice_list[self.session_col] \
            .map(str)  # sessionを文字列に変換

        # camera_colがself.mice_listにない場合はエラーを出す
        if camera_col not in self.mice_list:
            raise ValueError(f"`{camera_col}` is not in the mice list.")
        self.camera_col: str = camera_col
        self.mice_list[self.camera_col] = self.mice_list[self.camera_col] \
            .map(str)  # cameraを文字列に変換


class ExperimentFile:
    """AR-LABOのシステムが出力したファイルのメタデータを管理するクラス"""
    def __init__(self, file_path: str, fps: int):
        self.file_path: str = file_path
        self.fps: int = fps
        self.skip_rows: int = 0  # 座標データが始まる前のヘッダ部分の行数
        self.metadata: Dict[str, str] = {}

        # 出力ファイルに変な文字が入っておりerrorが出るため、errors=ignoreにする
        with open(self.file_path, "r", encoding="utf-8", errors="ignore") as f:
            current_line = f.readline()
            self.skip_rows += 1
            if current_line.strip() != "--- Analysis Data ---":
                raise ValueError("Invalid file format for AR-LABO CSV file")
            while True:
                current_line = f.readline()
                self.skip_rows += 1
                if current_line.strip() == "Experiment note":
                    break
                key, value = current_line.split(",", 1)
                self.metadata[key] = value.strip()
            while current_line.strip() != "***** Analysis Data *****":
                current_line = f.readline()
                self.skip_rows += 1

        # metadataに"Session Name"と"Camera No"が含まれているか確認する
        if "Session Name" not in self.metadata:
            raise ValueError("Invalid file format for AR-LABO CSV file")
        if "Camera No" not in self.metadata:
            raise ValueError("Invalid file format for AR-LABO CSV file")
        self.session: str = self.metadata["Session Name"]
        self.camera: str = self.metadata["Camera No"].split("@")[0].strip()

        # metadataに"AR Marker"が含まれているか確認する
        if "AR Marker" not in self.metadata:
            raise ValueError("Invalid file format for AR-LABO CSV file")
        self.mice: Dict[int, Dict[str, str]] = {}
        for i, pair in \
                enumerate(self.metadata["AR Marker"].split(","), start=1):
            marker_id, description = pair.split(":")
            self.mice[i] = {"marker_id": marker_id, "description": description}
        self.n_mice: int = len(self.mice)

    def mice_info(self, proj: Optional[Project] = None) -> pd.DataFrame:
        """マウスの情報をDataFrameで返す"""
        df = pd.DataFrame(columns=["id", "marker_id", "description"],
                          data=[[id, val["marker_id"], val["description"]]
                                for id, val in self.mice.items()])
        df = df.assign(session=self.session, camera=self.camera)
        if proj:
            df = df \
                .merge(proj.mice_list,
                       left_on=["marker_id", "session", "camera"],
                       right_on=[proj.marker_id_col,
                                 proj.session_col,
                                 proj.camera_col],
                       how="left") \
                .drop(columns=[proj.marker_id_col,
                               proj.session_col,
                               proj.camera_col])
        return df

    def file_name(self) -> str:
        """ファイル名を返す"""
        return self.file_path.split("/")[-1]


def generate_mice_list(data_dir: Path) -> pd.DataFrame:
    """data_dir内のCSVファイルから、マウスのリストを作成する"""
    list_df = pd.DataFrame()
    for csv_file in data_dir.glob("*.csv"):
        name = str(csv_file).split("/")[-1]
        ex = ExperimentFile(csv_file, fps=20)
        for k, v in ex.mice.items():
            list_df = pd.concat([list_df, pd.DataFrame({
                "Session": ex.session,
                "Camera": ex.camera,
                "ID": v["marker_id"],
                "Line": v["description"],
                "File": name
            }, index=[k])])
    return list_df


class Behavior:
    """マウスの時系列座標データを管理するクラス"""
    def read_raw(experiment: ExperimentFile) -> pd.DataFrame:
        # 変な文字が入っておりそのままread_csvするとskip_rowsしてもerrorが出る
        # ファイルバッファを取得して、skip_rows分だけ読み飛ばしてから渡す
        with open(experiment.file_path,
                  "r", encoding="utf-8", errors="ignore") as f:
            for _ in range(experiment.skip_rows):
                f.readline()
            # 1列目の一番下が"*************************"という文字列になっている
            # DtypeWarning回避のため、1列目（"Project No"）のdtypeをstrにする
            return pd.read_csv(f, dtype={"Project No": str})

    def experiment_id(self) -> str:
        """実験のIDを返す"""
        return f"{self.session}c{self.camera}"

    def __init__(self, experiment: ExperimentFile, rm_jump: bool = True):
        self.experiment = experiment
        self.fps: int = experiment.fps
        self.session: str = experiment.session
        self.camera: str = experiment.camera
        self.mice: Dict[int, Dict[str, str]] = experiment.mice
        self.n_mice: int = experiment.n_mice

        self.df: pd.DataFrame = Behavior.read_raw(experiment)

        # 座標のpxとmm単位の変換比率を計算する
        self.x_px_to_mm = \
            (self.df["ID-1 [Marker-X(mm)]"] / self.df["ID-1 [Marker-X]"]) \
            .value_counts().index[0]
        self.y_px_to_mm = \
            (self.df["ID-1 [Marker-Y(mm)]"] / self.df["ID-1 [Marker-Y]"]) \
            .value_counts().index[0]

        # date timeをtimeにリネームし、datetime型に変換する
        self.df = self.df.rename(columns={"date time": "time"})\
            .assign(time=lambda df: pd.to_datetime(df["time"]))

        # 不要な列を削除する。
        self.df = self.df.drop(
            columns=[col for col in self.df.columns
                     if ("Project No" in col or "Session No" in col or
                         "Camera No" in col or
                         "[Marker-X]" in col or "[Marker-Y]" in col or
                         "[Center-X]" in col or "[Center-Y]" in col or
                         "[Nose-X]" in col or "[Nose-Y]" in col or
                         "[Tail-X]" in col or "[Tail-Y]" in col or
                         "[Area]" in col)]
            )
        # 不要な列を削除する。
        self.df = self.df.drop(columns=([f"ID-{k}"
                                         for k in range(1, self.n_mice + 1)]))
        # 不要な列を削除する。
        self.df = self.df.drop(columns=([f"ID-{k} Name"
                                         for k in range(1, self.n_mice + 1)]))

        for k in range(1, self.n_mice + 1):
            # もし"ID-k [Angle]"列があれば"id_k_angle"にリネームする
            if f"ID-{k} [Angle]" in self.df.columns:
                self.df = self.df.rename(
                    columns={f"ID-{k} [Angle]": f"id_{k}_angle"})

            # ID-k [Marker-X(mm)]列が存在しなければ例外を出す
            if f"ID-{k} [Marker-X(mm)]" not in self.df.columns:
                raise ValueError(
                    f"`ID-{k} [Marker-X(mm)]` is not in the CSV file"
                )
            # ID-k [Marker-Y(mm)]列が存在しなければ例外を出す
            if f"ID-{k} [Marker-Y(mm)]" not in self.df.columns:
                raise ValueError(
                    f"`ID-{k} [Marker-Y(mm)]` is not in the CSV file"
                )

            # "id_k_x", "id_k_y"にリネームする
            self.df = self.df.rename(
                columns={f"ID-{k} [Marker-X(mm)]": f"id_{k}_x",
                         f"ID-{k} [Marker-Y(mm)]": f"id_{k}_y"}
            )

            # id_k_xとid_k_yの欠損値の位置が一致していなければ例外を出す
            # if not self.df[f"id_{k}_x"].isna().equals(
            #         self.df[f"id_{k}_y"].isna()):
            #     raise ValueError(f"id_{k}_x and id_{k}_y have different "
            #                      "missing values")

            # id_k_xがすべて欠損値なら例外を出す
            if self.df[f"id_{k}_x"].isna().all():
                raise ValueError(f"id_{k}_x is all missing values")
            # id_k_yがすべて欠損値なら例外を出す
            if self.df[f"id_{k}_y"].isna().all():
                raise ValueError(f"id_{k}_y is all missing values")
            self.df[f"id_{k}_missing"] = np.where(self.df[f"id_{k}_x"].isna(),
                                                  1, 0)

        # 各マウスで最初に有効な座標データが得られた行のmax（一番遅い時刻）
        self.start = max([self.df[f"id_{k}_x"].first_valid_index()
                          for k in range(1, self.n_mice + 1)])
        # startのindexに対応するtimeをstart_timeとする
        self.start_time = self.df["time"].iloc[self.start]
        # 各マウスで最後に有効な座標データが得られた行のmin（一番早い時刻）
        self.end = min([self.df[f"id_{k}_x"].last_valid_index()
                        for k in range(1, self.n_mice + 1)])
        # endのindexに対応するtimeをend_timeとする
        self.end_time = self.df["time"].iloc[self.end]
        # 有効な座標データが得られた時間
        self.duration: datetime.timedelta = self.end_time - self.start_time
        # 1秒間当たりのフレーム数（データの行数）を計算する
        self.actual_fps = \
            (self.end - self.start) / self.duration.total_seconds()

        # id_kの座標の欠損率をそれぞれ求める
        self.missing_rate = {
            f"id_{k}":
            self.df.loc[self.start:self.end, f"id_{k}_x"].isna().mean()
            for k in range(1, self.n_mice + 1)
        }

        # ケージの大きさを95パーセンタイルの座標の幅で推定する
        x_95tile = self.df[[f"id_{k}_x"
                            for k in range(1, self.n_mice + 1)]] \
                       .stack().quantile([0.025, 0.975])
        self.x_width_est = x_95tile[0.975] - x_95tile[0.025]
        y_95tile = self.df[[f"id_{k}_y"
                            for k in range(1, self.n_mice + 1)]] \
                       .stack().quantile([0.025, 0.975])
        self.y_width_est = y_95tile[0.975] - y_95tile[0.025]

        self.jump_index = {}

        # jumpが発生した場合、その行のid_kの座標を欠損させる
        if rm_jump:
            for k in range(1, self.n_mice + 1):
                temp_df = self.df[[f"id_{k}_x", f"id_{k}_y"]].copy()
                # 座標が欠損している行を削除
                temp_df = temp_df.dropna()
                # 1フレーム前からの移動距離を計算する
                temp_df[f"id_{k}_move"] = np.sqrt(
                    (temp_df[f"id_{k}_x"] - temp_df[f"id_{k}_x"].shift(1))**2 +
                    (temp_df[f"id_{k}_y"] - temp_df[f"id_{k}_y"].shift(1))**2
                )
                # 1フレーム前からの移動した角度を計算する
                temp_df[f"id_{k}_move_angle"] = np.arctan2(
                    temp_df[f"id_{k}_y"] - temp_df[f"id_{k}_y"].shift(1),
                    temp_df[f"id_{k}_x"] - temp_df[f"id_{k}_x"].shift(1)
                ) * 180 / np.pi
                # jump(1)が発生した場合、1、それ以外は0
                temp_df[f"id_{k}_jump_1"] = \
                    np.where(
                        (temp_df[f"id_{k}_move"] > 20) &
                        (temp_df[f"id_{k}_move"].shift(-1) > 20) &
                        ((temp_df[f"id_{k}_move_angle"].diff(-1).abs() > 165) &
                         (temp_df[f"id_{k}_move_angle"].diff(-1).abs() < 195)),
                        1,
                        0
                    )

                # jump(2)が発生した場合、2、それ以外は0
                temp_df[f"id_{k}_jump_2"] = \
                    np.where(
                        (temp_df[f"id_{k}_move"] > 20) &
                        (temp_df[f"id_{k}_move"].shift(-2) > 20) &
                        ((temp_df[f"id_{k}_move_angle"].diff(-2).abs() > 165) &
                         (temp_df[f"id_{k}_move_angle"].diff(-2).abs() < 195)),
                        2,
                        0
                    )
                temp_df[f"id_{k}_jump_2_2"] = \
                    np.where(
                        temp_df[f"id_{k}_jump_2"].shift(1) == 2,
                        2,
                        0
                    )

                # jumpが発生した行のindexをself.jump_indexに追加する
                self.jump_index[f"id_{k}"] = \
                    temp_df[((temp_df[f"id_{k}_jump_1"] == 1)
                             | (temp_df[f"id_{k}_jump_2"] == 2)
                             | (temp_df[f"id_{k}_jump_2_2"] == 2))].index

                # jumpが発生した座標を記録しておく
                id_k_x = temp_df.loc[self.jump_index[f'id_{k}'], f'id_{k}_x']
                id_k_y = temp_df.loc[self.jump_index[f'id_{k}'], f'id_{k}_y']
                self.df.loc[self.jump_index[f"id_{k}"], f"id_{k}_missing"] = \
                    self.jump_index[f"id_{k}"].map(
                        lambda x: f"({id_k_x[x]:.3f}, {id_k_y[x]:.3f})"
                    )

                # jumpが発生した行のid_kの座標をself.dfから削除する
                self.df.loc[self.jump_index[f"id_{k}"],
                            [f"id_{k}_x", f"id_{k}_y"]] = np.nan
            del temp_df

        # id_k_x, id_k_yを線形補間する
        for k in range(1, self.n_mice + 1):
            self.df[f"id_{k}_x"] = self.df[f"id_{k}_x"] \
                .interpolate(method="linear")
            self.df[f"id_{k}_y"] = self.df[f"id_{k}_y"] \
                .interpolate(method="linear")

        # start_indexからend_indexまでの行のみ残す（先に線形補間するのを忘れない）
        self.df = self.df.loc[self.start:self.end]

        # 一番上の時刻を0とした相対時刻time0を追加する
        self.df = self.df.assign(time0=(self.df["time"]
                                        - self.df["time"].iloc[0]))
        # time0を秒に変換した列を追加する
        self.df = self.df.assign(time0_sec=self.df["time0"].dt.total_seconds())

        # k, j (k, jは1~self.n_mice) の全組み合わせについて、dist_k_jを計算する
        for k in range(1, self.n_mice + 1):
            for j in range(k + 1, self.n_mice + 1):
                self.df[f"dist_{k}_{j}"] = np.sqrt(
                    (self.df[f"id_{k}_x"] - self.df[f"id_{j}_x"])**2 +
                    (self.df[f"id_{k}_y"] - self.df[f"id_{j}_y"])**2
                )

        # id_k (kは1~self.n_mice) について、1フレーム前からの移動距離を計算する
        for k in range(1, self.n_mice + 1):
            self.df[f"id_{k}_move"] = np.sqrt(
                (self.df[f"id_{k}_x"] - self.df[f"id_{k}_x"].shift(1))**2 +
                (self.df[f"id_{k}_y"] - self.df[f"id_{k}_y"].shift(1))**2
            )
        # id_k (kは1~self.n_mice) について、1フレーム前からの移動した角度を計算する
        for k in range(1, self.n_mice + 1):
            self.df[f"id_{k}_move_angle"] = np.arctan2(
                self.df[f"id_{k}_y"] - self.df[f"id_{k}_y"].shift(1),
                self.df[f"id_{k}_x"] - self.df[f"id_{k}_x"].shift(1)
            ) * 180 / np.pi

        # id_k (kは1~self.n_mice) について、1フレーム前からの移動距離が10mm以上で
        # かつ移動の角度の変化が160～195度の場合、1、それ以外は0
        for k in range(1, self.n_mice + 1):
            self.df[f"id_{k}_move_turning"] = \
                np.where(
                    (self.df[f"id_{k}_move"] > 25) &
                    (self.df[f"id_{k}_move"].shift(-1) > 25) &
                    ((self.df[f"id_{k}_move_angle"].diff(-1).abs() > 165) &
                     (self.df[f"id_{k}_move_angle"].diff(-1).abs() < 195)),
                    1,
                    0
                )

        # id_k (kは1~self.n_mice) について、
        # マウスの直前の1/2秒間の移動したベクトルの角度（0～360度）を計算する
        for k in range(1, self.n_mice + 1):
            self.df[f"id_{k}_angle"] = np.arctan2(
                self.df[f"id_{k}_y"]
                - self.df[f"id_{k}_y"].shift(self.fps // 4),
                self.df[f"id_{k}_x"]
                - self.df[f"id_{k}_x"].shift(self.fps // 4)
            ) * 180 / np.pi
            # 1/2秒前からのid_k_moveの和が5mm未満の場合、id_k_angleをnanにする
            self.df.loc[
                self.df[f"id_{k}_move"]
                    .shift(self.fps // 4).rolling(self.fps // 4).sum() < 5,
                f"id_{k}_angle"
            ] = np.nan
            # 欠損させた値はffillとbfillで補完する
            self.df[f"id_{k}_angle"] = \
                self.df[f"id_{k}_angle"] \
                    .fillna(method="ffill").fillna(method="bfill")


class Report:
    def __init__(self, behavior: Behavior, project: Project):
        behavior = behavior
        experiment = behavior.experiment
        project = project

        self.file_path = experiment.file_path
        self.session = behavior.session
        self.camera = behavior.camera
        self.experiment_id = f"{self.session}c{self.camera}"
        self.duration = behavior.duration
        self.start = behavior.start
        self.end = behavior.end
        self.frames = self.end - self.start
        self.actual_fps = behavior.actual_fps
        self.x_px_to_mm = behavior.x_px_to_mm
        self.y_px_to_mm = behavior.y_px_to_mm
        self.x_width_est = behavior.x_width_est
        self.y_width_est = behavior.y_width_est
        self.project_info = project.mice_list[
            (project.mice_list["Session"] == self.session)
            & (project.mice_list["Camera"] == self.camera)]
        self.mice = behavior.mice
        self.n_mice = len(self.mice)
        self.mice_info = ", ".join(
            [f"id_{k}: #{v['marker_id']}({v['description']})"
             for k, v in self.mice.items()])
        self.missing_rate = behavior.missing_rate.copy()
        self.missing = ", ".join(
            [f"{k}: {v * 100:.2f}%" for k, v in self.missing_rate.items()])
        self.jump_index = behavior.jump_index.copy()
        self.jump = ", ".join(
            [f"{k}: {len(v)} ("
             f"{len(v) / (self.end - self.start) * 100:.2f}"
             "%)" for k, v in self.jump_index.items()])

    def __str__(self) -> str:
        report = []
        report.append("=" * 80)
        report.append(f"Analysis of {self.file_path}:")
        report.append(f" [Session] {self.session}, [Camera] {self.camera}")
        report.append(f" [Rec Duration] {self.duration}")
        report.append(f" [Total Frames] {self.frames},"
                      f" [Actual FPS] {self.actual_fps:.2f}")
        report.append(f" [Ratio mm/px] x={self.x_px_to_mm:.3f},"
                      f" y={self.y_px_to_mm:.3f}")
        report.append(f" [Est Cage Size] x={self.x_width_est:.1f}mm,"
                      f" y={self.y_width_est:.1f}mm")
        report.append(" " + "-" * 79)
        report.append(" [Project Info]\n "
                      + self.project_info
                      .to_string(index=False, index_names=False, line_width=79)
                      .replace("\n", "\n "))
        report.append(" " + "-" * 79)
        report.append(f" [Mouse] N={self.n_mice}; {self.mice_info}")
        report.append(f" [Missing] {self.missing}")
        report.append(f" [Jump] {self.jump}")
        return "\n".join(report)

    def __repr__(self) -> str:
        return self.__str__()

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "experiment_id": [self.experiment_id],
            "file_path": [self.file_path],
            "session": [self.session],
            "camera": [self.camera],
            "duration": [self.duration],
            "frames": [self.frames],
            "actual_fps": [self.actual_fps],
            "x_px_to_mm": [self.x_px_to_mm],
            "y_px_to_mm": [self.y_px_to_mm],
            "x_width_est": [self.x_width_est],
            "y_width_est": [self.y_width_est],
            "project_info": [self.project_info.to_string(index=False,
                                                         index_names=False)],
            "n_mice": [self.n_mice],
            "mice_info": [self.mice_info],
            "missing_rate": [self.missing],
            "jump_index": [self.jump]
        })

    def warnings(self,
                 duration=None,
                 missing_rate_threshold=0.85,
                 jump_rate_threshold=0.1) -> list[str]:
        warnings = []
        if duration is not None and self.duration < duration:
            warnings.append(f" NOT ENOUGH TIME FOR ANALYSIS: "
                            f"{self.duration} < {duration}")
        for k, v in self.missing_rate.items():
            if v > missing_rate_threshold:
                warnings.append(f" MISSING RATE TOO HIGH: {k}, {v * 100:.2f}%")
        for k, v in self.jump_index.items():
            if len(v) / (self.end - self.start) * 100 > jump_rate_threshold:
                warnings.append(f" TOO MUCH JUMP ERROR: {k}, {len(v)}")
        return warnings


class Interaction:
    """閾値を持ったマウス同士の相互作用を計算する"""
    def __init__(self, behavior: Behavior, threshold_dist: int):
        self.threshold_dist: int = threshold_dist
        self.df: pd.DataFrame = behavior.df.copy()
        self.fps: int = behavior.fps
        self.session: str = behavior.session
        self.camera: str = behavior.camera
        self.mice:  Dict[int, Dict[str, str]] = behavior.mice
        self.n_mice: int = behavior.n_mice

        for k in range(1, behavior.n_mice + 1):
            for j in range(k + 1, behavior.n_mice + 1):
                # マウス間距離が`threshold_dist`未満なら1、それ以外は0
                self.df[f"dist_{k}_{j}_nearing"] = \
                    np.where(self.df[f"dist_{k}_{j}"] < threshold_dist, 1, 0)

                # マウス同士が近づくevent発生で-1、離れるevent発生で1、それ以外は0
                self.df[f"dist_{k}_{j}_event_type"] = \
                    np.where(
                        self.df[f"dist_{k}_{j}_nearing"].diff(1) == -1,
                        1,
                        np.where(
                            self.df[f"dist_{k}_{j}_nearing"].diff(1) == 1,
                            -1,
                            0
                        )
                    )

                # eventが発生した時刻を記録する。それ以外の行は欠損値
                self.df[f"dist_{k}_{j}_event_time"] = \
                    np.where(self.df[f"dist_{k}_{j}_event_type"] != 0,
                             self.df["time0_sec"], np.nan)

                # eventが発生した際に次のeventまでの時間を計算する。
                # eventのない行は欠損値
                self.df[f"dist_{k}_{j}_event_duration"] = \
                    self.df[f"dist_{k}_{j}_event_time"] \
                        .fillna(method="bfill").diff().shift(-1) \
                        .replace(0, np.nan)

            # 移動距離のself.fpsフレーム分の移動和を計算する
            self.df[f"id_{k}_move_rolling_R"] = \
                self.df[f"id_{k}_move"].rolling(self.fps).sum()
            self.df[f"id_{k}_move_rolling_F"] = \
                self.df[f"id_{k}_move_rolling_R"].shift(-self.fps)


# 実験条件のうち時間の条件を管理するためのクラスとヘルパー関数群

def try_int(value: Union[datetime.timedelta, int, float]) -> Union[int, float]:
    """valueをintに変換できるならintに変換し、できないならそのまま返す"""
    if isinstance(value, datetime.timedelta):
        value = value.total_seconds()
    if np.isclose(value, int(value), rtol=0):
        return int(value)
    else:
        return value


def ensure_timedelta(value: Union[datetime.timedelta, int, float]
                     ) -> datetime.timedelta:
    """valueがdatetime.timedelta型ならそのまま返し、
    valueがintやfloatならばdatetime.timedeltaに変換する"""
    if isinstance(value, datetime.timedelta):
        return value
    elif isinstance(value, (int, float)):
        return datetime.timedelta(seconds=value)
    else:
        raise TypeError(
            f"value must be timedelta, int or float, got {type(value)}"
        )


class TimeClip:
    """clipの時間のstart, duration, endを管理するクラス"""
    def __init__(self,
                 start: Union[datetime.timedelta, int, float]
                 = datetime.timedelta(0),
                 duration: Optional[Union[datetime.timedelta, int, float]]
                 = None,
                 end: Optional[Union[datetime.timedelta, int, float]] = None):
        if isinstance(start, datetime.timedelta):
            self.start = start
        else:
            self.start = datetime.timedelta(seconds=start)
        if duration:
            if end:
                raise ValueError(
                    "duration and end cannot be specified at the same time")
            if isinstance(duration, datetime.timedelta):
                self.duration = duration
            else:
                self.duration = datetime.timedelta(seconds=duration)
            if self.duration < datetime.timedelta(0):
                raise ValueError("duration must be positive")
            self.end = self.start + self.duration

        elif end:
            if isinstance(end, datetime.timedelta):
                self.end = end
            else:
                self.end = datetime.timedelta(seconds=end)
            if self.end < self.start:
                raise ValueError("end must be greater than start")
            self.duration = self.end - self.start
        else:
            raise ValueError("duration or end must be specified")

    def __repr__(self) -> str:
        return f"TimeClip(start={self.start}, duration={self.duration}, " \
               f"end={self.end})"

    def __str__(self) -> str:
        return f"{self.start.total_seconds()} - {self.end.total_seconds()} sec"

    def __eq__(self, other: "TimeClip") -> bool:
        return self.start == other.start and self.duration == other.duration

    def __lt__(self, other: "TimeClip") -> bool:
        return self.end < other.start

    def __le__(self, other: "TimeClip") -> bool:
        return self.end <= other.start

    def __gt__(self, other: "TimeClip") -> bool:
        return self.start > other.end

    def __ge__(self, other: "TimeClip") -> bool:
        return self.start >= other.end

    def __contains__(self, other: Union[datetime.timedelta, int, float]) \
            -> bool:
        # endは含まない
        return (self.start <= datetime.timedelta(seconds=other)
                and datetime.timedelta(seconds=other) < self.end)

    def __len__(self) -> float:
        return self.duration.total_seconds()

    def __hash__(self) -> int:
        return hash((self.start, self.duration, self.end))

    def __bool__(self) -> bool:
        return bool(self.duration)

    @staticmethod
    def from_datetime(time0_from: Union[Behavior, Interaction,
                                        datetime.datetime],
                      start: Optional[Union[datetime.datetime,
                                            datetime.timedelta,
                                            int, float]] = 0,
                      duration: Optional[Union[datetime.timedelta,
                                               int, float]] = None,
                      end: Optional[Union[datetime.datetime,
                                          datetime.timedelta,
                                          int, float]] = None
                      ) -> "TimeClip":
        """time0を基準にした時刻ではなく現実のdatetimeを使う場合"""
        if isinstance(time0_from, (Behavior, Interaction)):
            # time0 == 0 に対応するtime
            time0_start = time0_from.df["time"].iloc[0]
        elif isinstance(time0_from, datetime.datetime):
            time0_start = time0_from
        else:
            raise TypeError("time0_from must be Behavior, Interaction "
                            "or datetime.datetime")

        if isinstance(start, datetime.datetime):
            start: datetime.timedelta = start - time0_start
        elif isinstance(start, datetime.timedelta):
            pass
        elif isinstance(start, str):
            try:
                start: datetime.timedelta = \
                    datetime.datetime.fromisoformat(start) - time0_start
            except ValueError:
                try:
                    date: datetime.date = time0_start.date()
                    time: datetime.time = datetime.time.fromisoformat(start)
                    start: datetime.timedelta = \
                        datetime.datetime.combine(date, time) - time0_start
                except ValueError:
                    raise ValueError("start must be isoformat when str")

        else:
            start: datetime.timedelta = datetime.timedelta(seconds=start)

        if duration:
            if end:
                raise ValueError("duration and end cannot be specified "
                                 "at the same time")
            if isinstance(duration, datetime.timedelta):
                pass
            else:
                duration: datetime.timedelta = \
                    datetime.timedelta(seconds=duration)
            return TimeClip(start=start, duration=duration)
        elif end:
            if isinstance(end, datetime.datetime):
                end: datetime.timedelta = end - time0_start
            elif isinstance(end, datetime.timedelta):
                pass
            elif isinstance(end, str):
                try:
                    end: datetime.timedelta = \
                        datetime.datetime.fromisoformat(end) - time0_start
                except ValueError:
                    try:
                        date: datetime.date = time0_start.date()
                        time: datetime.time = datetime.time.fromisoformat(end)
                        end: datetime.timedelta = \
                            datetime.datetime.combine(date, time) - time0_start
                    except ValueError:
                        raise ValueError("end must be isoformat when str")

            else:
                end: datetime.timedelta = datetime.timedelta(seconds=end)
            return TimeClip(start=start, end=end)
        else:
            if isinstance(time0_from, (Behavior, Interaction)):
                end = time0_from.df["time"].iloc[-1] - time0_start \
                    + datetime.timedelta(milliseconds=1)
                return TimeClip(start=start, end=end)

            raise ValueError("duration or end must be specified")


class TimeClipList:
    """複数のTimeClipを保持するクラス"""
    def __init__(self,
                 time_clips: Union[None, TimeClip, List[TimeClip]] = None,
                 labels: Optional[Union[str, List[str]]] = None,
                 check_ascending_order: bool = True):
        self.time_clips: List[TimeClip] = []
        self.labels: List[str] = []
        self.add(time_clips, labels, check_ascending_order)

    def __repr__(self) -> str:
        return f"TimeClipList({self.time_clips}, {self.labels})"

    def __str__(self) -> str:
        if len(self) == 0:
            return "<Empty TimeClipList>"
        else:
            return "\n".join([f"{label}: {time_clip}" for label, time_clip
                              in zip(self.labels, self.time_clips)])

    def __len__(self) -> int:
        return len(self.time_clips)

    def add(self,
            time_clips: Union[None, TimeClip, List[TimeClip]] = None,
            labels: Optional[Union[str, List[str]]] = None,
            check_ascending_order: bool = True) -> "TimeClipList":

        if time_clips is not None:
            if isinstance(time_clips, TimeClip):
                time_clips: List[TimeClip] = [time_clips]
            # time_clipsが昇順に並んでいることをチェック
            if check_ascending_order:
                if any([time_clips[i] > time_clips[i + 1] for i
                        in range(len(time_clips) - 1)]):
                    raise ValueError(
                        "time_clips must be sorted in ascending order")
        else:
            time_clips: List[TimeClip] = []

        if labels:
            if isinstance(labels, str):
                labels: List[str] = [labels]
            if len(time_clips) != len(labels):
                raise ValueError(
                    "length of time_clips and labels must be same")
        else:
            labels: List[str] = [f"time_clip_{len(self) + i}"
                                 for i in range(len(time_clips))]

        self.time_clips.extend(time_clips)
        self.labels.extend(labels)

        # labelsがuniqueかどうかチェック
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("labels must be unique")

        return self

    def remove(self, index: int) -> "TimeClipList":
        del self.time_clips[index]
        del self.labels[index]
        return self

    def remove_by_label(self, label: str) -> "TimeClipList":
        self.remove(self.labels.index(label))
        return self

    def add_bins(self,
                 bin_width: Union[datetime.timedelta, int, float],
                 timeclip: TimeClip,
                 labels: Optional[Union[str, List[str]]] = None,
                 label_generator: Optional[callable] = None,
                 ) -> "TimeClipList":
        bin_width: datetime.timedelta = ensure_timedelta(bin_width)

        time_clips: List[TimeClip] = []
        generate_label: bool = False
        if labels is None:
            generate_label = True
            labels = []

        for bin_n in range(timeclip.duration // bin_width +
                           (0 if (timeclip.duration % bin_width ==
                                  datetime.timedelta(seconds=0)) else 1)):
            bin = timeclip.start + bin_n * bin_width
            time_clips.append(
                TimeClip(start=bin,
                         duration=min(bin_width, timeclip.end - bin))
            )
            if generate_label:
                if label_generator is None:
                    labels.append(f"bin{try_int(bin)}")
                else:
                    labels.append(label_generator(timeclip, bin))

        self.add(time_clips, labels)

        return self

    def duration(self) -> datetime.timedelta:
        if len(self) > 0:
            return self.time_clips[-1].end - self.time_clips[0].start
        else:
            return datetime.timedelta(0)

    def __iter__(self) -> Iterator[Tuple[str, TimeClip]]:
        return zip(self.labels, self.time_clips)


def label_hhmm(timeclip: TimeClip,
               bin: datetime.timedelta,
               start_as: Union[datetime.timedelta, str] = "00:00"
               ) -> str:
    """timeclipとbinとstart_asを受け取って13:00のようなlabelを返す関数。
    TimeClipListにadd_binsする際に使用する。"""
    if isinstance(start_as, str):
        _time = datetime.datetime.strptime(start_as, "%H:%M")
        start_as = datetime.timedelta(hours=_time.hour, minutes=_time.minute)
    diff = bin - timeclip.start
    t = start_as + diff
    # "13:00"のような形式で返す
    return (f"{try_int(t.total_seconds()) // 3600:02}:"
            f"{try_int(t.total_seconds()) % 3600 // 60:02}")


def label_binmm(timeclip: TimeClip,
                bin: datetime.timedelta) -> str:
    """timeclipとbinを受け取って、bin30のようなlabelを返す関数。
    TimeClipListにadd_binsする際に使用する。"""
    diff = bin - timeclip.start
    # "bin30"のような形式で返す
    return f"bin{try_int(diff.total_seconds() // 60)}"


def clip_behavior_df(behavior: Union[Behavior, Interaction],
                     timeclip: Optional[TimeClip] = None
                     ) -> Tuple[pd.DataFrame, TimeClip]:
    """Behaviorからtimeclipに相当する範囲を切り出す"""
    if timeclip is None:
        start = behavior.df["time0"].iloc[0]
        # 1ミリセカンド後の時刻をendとする（endには等号がつかないためずらす必要）
        end = behavior.df["time0"].iloc[-1] \
            + datetime.timedelta(milliseconds=1)
        return behavior.df, TimeClip(start=start, end=end)
    elif isinstance(timeclip, TimeClip):
        df = behavior.df.loc[(behavior.df["time0"] >= timeclip.start) &
                             (behavior.df["time0"] < timeclip.end)]
        return df, timeclip
    else:
        raise ValueError("timeclip must be TimeClip or None")


def draw_trajectory(behavior: Behavior,
                    timeclip: Optional[TimeClip] = None,
                    draw_ids: Optional[List[Union[str, int]]] = None,
                    scatter: bool = False) -> plt.Figure:
    """指定したtimeclipの範囲で軌跡を描画する関数"""
    xlim = (min([behavior.df[f"id_{k}_x"].min()
                 for k in range(1, behavior.n_mice + 1)]),
            max([behavior.df[f"id_{k}_x"].max()
                 for k in range(1, behavior.n_mice + 1)]))
    ylim = (min([behavior.df[f"id_{k}_y"].min()
                 for k in range(1, behavior.n_mice + 1)]),
            max([behavior.df[f"id_{k}_y"].max()
                 for k in range(1, behavior.n_mice + 1)]))

    df, timeclip = clip_behavior_df(behavior, timeclip)

    title = f"Session: {behavior.session}, Camera: {behavior.camera}" \
            f" [Time0: {timeclip}]"

    # マウスの軌跡のグラフを同じスケールで描く
    fig = plt.figure()

    ax = fig.add_subplot(1, 1, 1, aspect="equal")
    for k in range(1, behavior.n_mice + 1):
        if draw_ids is None or f"{k}" in draw_ids or k in draw_ids:
            if not scatter:
                ax.plot(df[f"id_{k}_x"], df[f"id_{k}_y"],
                        alpha=0.7, lw=0.2, label=f"ID-{k}")
            else:
                ax.scatter(df[f"id_{k}_x"], df[f"id_{k}_y"],
                           alpha=0.7, label=f"ID-{k}", s=0.05)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_ylim(ax.get_ylim()[::-1])  # y軸を上下逆にする
        ax.set_title(title)

    # 凡例をグラフの外に表示
    leg = ax.legend(bbox_to_anchor=(1, 1), loc='upper left')

    if not scatter:
        # レジェンドの線の幅を太くする
        for line in leg.get_lines():
            line.set_linewidth(2)
    else:
        # レジェンドの点のalphaを1にし、点のサイズを5にする
        for lh in leg.legendHandles:
            lh.set_alpha(1)
            lh._sizes = [5]

    plt.close()  # Jupyter Notebook上で意図せず表示しないために閉じる
    return fig


def merge_project(result: pd.DataFrame, proj: Optional[Project]) \
        -> pd.DataFrame:
    """解析結果に実験の情報をmergeするヘルパー関数"""
    if not isinstance(result, pd.DataFrame):
        raise TypeError(f"result must be pandas.DataFrame, got {type(result)}")

    if not all([col in result.columns
                for col in ["marker_id", "session", "camera"]]):
        raise ValueError(
            "result must have columns 'marker_id', 'session', 'camera'")

    if isinstance(proj, Project):
        return result \
            .merge(proj.mice_list,
                   left_on=["marker_id", "session", "camera"],
                   right_on=[proj.marker_id_col,
                             proj.session_col,
                             proj.camera_col],
                   how="left") \
            .drop(columns=[proj.marker_id_col,
                           proj.session_col,
                           proj.camera_col])
    elif proj is None:
        return result
    else:
        raise TypeError(f"proj must be Project or None, got {type(proj)}")


def calc_activity(behavior: Behavior,
                  timeclips: Optional[Union[TimeClipList, TimeClip]] = None,
                  proj: Optional[Project] = None
                  ) -> pd.DataFrame:
    """各マウスのtimeclips毎の移動距離（活動量）を計算する"""
    if timeclips is None:
        _, timeclips = clip_behavior_df(behavior)
    if isinstance(timeclips, TimeClip):
        timeclips = TimeClipList(timeclips, labels="total")

    activity = pd.DataFrame()
    for label, timeclip in timeclips:
        clip_df, timeclip = clip_behavior_df(behavior, timeclip)
        for k in range(1, behavior.n_mice + 1):
            activity = pd.concat([activity, pd.DataFrame({
                "id": k,
                "duration": try_int(timeclips.duration()),
                "bin": label,
                "bin_width": try_int(timeclip.duration),
                "bin_start": try_int(timeclip.start),
                "bin_end": try_int(timeclip.end),
                "activity": clip_df[f"id_{k}_move"].sum() / 1000,
                "session": behavior.session,
                "camera": behavior.camera,
                "marker_id": behavior.mice[k]["marker_id"],
                "description": behavior.mice[k]["description"]
            }, index=[k])])

    return merge_project(activity, proj)


def calc_distance(behavior: Behavior,
                  timeclips: Optional[Union[TimeClipList, TimeClip]] = None,
                  proj: Optional[Project] = None,
                  calc_total: bool = True,
                  calc_total_as_long: bool = True,
                  calc_every_pair: bool = False,
                  ) -> pd.DataFrame:
    """各マウスのtimeclips毎の距離を計算する"""
    if timeclips is None:
        _, timeclips = clip_behavior_df(behavior)
    if isinstance(timeclips, TimeClip):
        timeclips = TimeClipList(timeclips, labels="total")

    distance = pd.DataFrame()
    for label, timeclip in timeclips:
        clip_df, timeclip = clip_behavior_df(behavior, timeclip)
        for k in range(1, behavior.n_mice + 1):
            # 各マウスについて、他のマウスとの距離を計算する
            # totalの計算結果をmean_distance列とclosest_distance列にする場合
            if calc_total and not calc_total_as_long:
                if calc_every_pair:
                    raise ValueError("calc_every_pair must be False "
                                     "when calc_total_as_long is False")
                distance = pd.concat([distance, pd.DataFrame({
                    "id": k,
                    "duration": try_int(timeclips.duration()),
                    "bin": label,
                    "bin_width": try_int(timeclip.duration),
                    "bin_start": try_int(timeclip.start),
                    "bin_end": try_int(timeclip.end),
                    "closest_distance": np.min(
                        [clip_df[f"dist_{min(k, j)}_{max(k, j)}"]
                            for j in range(1, behavior.n_mice + 1) if j != k],
                        axis=0
                    ).mean(),
                    "mean_distance": np.mean(
                        [clip_df[f"dist_{min(k, j)}_{max(k, j)}"]
                            for j in range(1, behavior.n_mice + 1) if j != k],
                        axis=0
                    ).mean(),
                    "session": behavior.session,
                    "camera": behavior.camera,
                    "marker_id": behavior.mice[k]["marker_id"],
                    "description": behavior.mice[k]["description"]
                }, index=[k])])
            # totalの計算結果をid_othersを"all_mean"と"closest"で区別する場合
            elif calc_total:
                # mean_distanceに相当する計算結果
                distance = pd.concat([distance, pd.DataFrame({
                    "id": k,
                    "id_other": "all_mean",
                    "duration": try_int(timeclips.duration()),
                    "bin": label,
                    "bin_width": try_int(timeclip.duration),
                    "bin_start": try_int(timeclip.start),
                    "bin_end": try_int(timeclip.end),
                    "distance": np.mean(
                        [clip_df[f"dist_{min(k, j)}_{max(k, j)}"]
                            for j in range(1, behavior.n_mice + 1) if j != k],
                        axis=0
                    ).mean(),
                    "session": behavior.session,
                    "camera": behavior.camera,
                    "marker_id": behavior.mice[k]["marker_id"],
                    "description": behavior.mice[k]["description"]
                }, index=[k])])

                # closest_distanceに相当する計算結果
                distance = pd.concat([distance, pd.DataFrame({
                    "id": k,
                    "id_other": "closest",
                    "duration": try_int(timeclips.duration()),
                    "bin": label,
                    "bin_width": try_int(timeclip.duration),
                    "bin_start": try_int(timeclip.start),
                    "bin_end": try_int(timeclip.end),
                    "distance": np.min(
                        [clip_df[f"dist_{min(k, j)}_{max(k, j)}"]
                            for j in range(1, behavior.n_mice + 1) if j != k],
                        axis=0
                    ).mean(),
                    "session": behavior.session,
                    "camera": behavior.camera,
                    "marker_id": behavior.mice[k]["marker_id"],
                    "description": behavior.mice[k]["description"]
                }, index=[k])])

            # 全ての組み合わせの距離を計算する
            if calc_every_pair:
                for j in range(1, behavior.n_mice + 1):
                    if k == j:
                        continue
                    distance = pd.concat([distance, pd.DataFrame({
                        "id": k,
                        "id_other": j,
                        "duration": try_int(timeclips.duration()),
                        "bin": label,
                        "bin_width": try_int(timeclip.duration),
                        "bin_start": try_int(timeclip.start),
                        "bin_end": try_int(timeclip.end),
                        "distance": np.mean(
                            clip_df[f"dist_{min(k, j)}_{max(k, j)}"]),
                        "session": behavior.session,
                        "camera": behavior.camera,
                        "marker_id": behavior.mice[k]["marker_id"],
                        "description": behavior.mice[k]["description"]
                    }, index=[k])])

    return merge_project(distance, proj)


def calc_interaction(inter: Interaction,
                     timeclips: Optional[Union[TimeClipList, TimeClip]] = None,
                     proj: Optional[Project] = None,
                     calc_total: bool = True,
                     calc_every_pair: bool = False
                     ) -> pd.DataFrame:
    """各マウスのtimeclips毎の接触回数・接触時間を計算する"""
    if timeclips is None:
        _, timeclips = clip_behavior_df(inter)
    if isinstance(timeclips, TimeClip):
        timeclips = TimeClipList(timeclips, labels="total")

    contact = pd.DataFrame()
    for label, timeclip in timeclips:
        clip_df, timeclip = clip_behavior_df(inter, timeclip)
        for k in range(1, inter.n_mice + 1):
            if calc_total:
                contact = pd.concat([contact, pd.DataFrame({
                    "id": k,
                    "id_other": "all_sum",
                    "duration": try_int(timeclips.duration()),
                    "bin": label,
                    "bin_width": try_int(timeclip.duration),
                    "bin_start": try_int(timeclip.start),
                    "bin_end": try_int(timeclip.end),

                    "contact_count": np.nansum(
                        [clip_df[f"dist_{min(k, j)}_{max(k, j)}_event_type"]
                         == -1 for j in range(1, inter.n_mice + 1) if j != k],
                        axis=0
                    ).sum(),
                    "approach_count": np.nansum(
                        [(clip_df[f"dist_{min(k, j)}_{max(k, j)}_event_type"]
                          == -1)
                         & (clip_df[f"id_{k}_move_rolling_R"]
                            >= clip_df[f"id_{j}_move_rolling_R"])
                         for j in range(1, inter.n_mice + 1) if j != k],
                        axis=0
                    ).sum(),
                    "receive_count": np.nansum(
                        [(clip_df[f"dist_{min(k, j)}_{max(k, j)}_event_type"]
                          == -1)
                         & (clip_df[f"id_{k}_move_rolling_R"]
                            < clip_df[f"id_{j}_move_rolling_R"])
                         for j in range(1, inter.n_mice + 1) if j != k],
                        axis=0
                    ).sum(),

                    "contact_duration": np.nansum(
                        [(clip_df[f"dist_{min(k, j)}_{max(k, j)}_event_type"]
                          == -1)
                         * clip_df[
                             f"dist_{min(k, j)}_{max(k, j)}_event_duration"]
                         for j in range(1, inter.n_mice + 1) if j != k],
                        axis=0
                    ).sum(),
                    "approach_duration": np.nansum(
                        [((clip_df[f"dist_{min(k, j)}_{max(k, j)}_event_type"]
                           == -1)
                          & (clip_df[f"id_{k}_move_rolling_R"]
                             >= clip_df[f"id_{j}_move_rolling_R"]))
                         * clip_df[
                             f"dist_{min(k, j)}_{max(k, j)}_event_duration"]
                         for j in range(1, inter.n_mice + 1) if j != k],
                        axis=0
                    ).sum(),
                    "receive_duration": np.nansum(
                        [((clip_df[f"dist_{min(k, j)}_{max(k, j)}_event_type"]
                           == -1)
                          & (clip_df[f"id_{k}_move_rolling_R"]
                             < clip_df[f"id_{j}_move_rolling_R"]))
                         * clip_df[
                             f"dist_{min(k, j)}_{max(k, j)}_event_duration"]
                         for j in range(1, inter.n_mice + 1) if j != k],
                        axis=0
                    ).sum(),

                    "session": inter.session,
                    "camera": inter.camera,
                    "marker_id": inter.mice[k]["marker_id"],
                    "description": inter.mice[k]["description"]
                }, index=[k])])
            if calc_every_pair:
                for j in range(1, inter.n_mice + 1):
                    if k == j:
                        continue

                    contact = pd.concat([contact, pd.DataFrame({
                        "id": k,
                        "id_other": j,
                        "duration": try_int(timeclips.duration()),
                        "bin": label,
                        "bin_width": try_int(timeclip.duration),
                        "bin_start": try_int(timeclip.start),
                        "bin_end": try_int(timeclip.end),

                        # マウスkがマウスjと接触した回数
                        "contact_count": np.sum(
                            clip_df[
                                f"dist_{min(k, j)}_{max(k, j)}_event_type"]
                            == -1
                        ),
                        # マウスkがマウスjにアプローチした回数
                        "approach_count": np.sum(
                            (clip_df[
                                f"dist_{min(k, j)}_{max(k, j)}_event_type"]
                             == -1)
                            & (clip_df[f"id_{k}_move_rolling_R"]
                               >= clip_df[f"id_{j}_move_rolling_R"])
                        ),
                        # マウスkがマウスjからアプローチされた回数
                        "receive_count": np.sum(
                            (clip_df[
                                f"dist_{min(k, j)}_{max(k, j)}_event_type"]
                             == -1)
                            & (clip_df[f"id_{k}_move_rolling_R"]
                               < clip_df[f"id_{j}_move_rolling_R"])
                        ),
                        # マウスkがマウスjと接触した時間
                        "contact_duration": np.nansum(
                            (clip_df[
                                f"dist_{min(k, j)}_{max(k, j)}_event_type"]
                             == -1)
                            * clip_df[
                                f"dist_{min(k, j)}_{max(k, j)}_event_duration"]
                        ),
                        # マウスkがマウスjにアプローチした時間
                        "approach_duration": np.nansum(
                            ((clip_df[
                                f"dist_{min(k, j)}_{max(k, j)}_event_type"]
                              == -1)
                             & (clip_df[f"id_{k}_move_rolling_R"]
                                >= clip_df[f"id_{j}_move_rolling_R"]))
                            * clip_df[
                                f"dist_{min(k, j)}_{max(k, j)}_event_duration"]
                        ),
                        # マウスkがマウスjからアプローチされた時間
                        "receive_duration": np.nansum(
                            ((clip_df[
                                f"dist_{min(k, j)}_{max(k, j)}_event_type"]
                              == -1)
                             & (clip_df[f"id_{k}_move_rolling_R"]
                                < clip_df[f"id_{j}_move_rolling_R"]))
                            * clip_df[
                                f"dist_{min(k, j)}_{max(k, j)}_event_duration"]
                        ),
                        "session": inter.session,
                        "camera": inter.camera,
                        "marker_id": inter.mice[k]["marker_id"],
                        "description": inter.mice[k]["description"]
                    }, index=[k])])

    return merge_project(contact, proj)


def calc_inter_entropy(inter: Interaction,
                       timeclips: Optional[Union[TimeClipList, TimeClip]]
                       = None,
                       proj: Optional[Project] = None
                       ) -> pd.DataFrame:
    """マウス間のtimeclips毎の接触回数・接触時間のエントロピーを計算する"""
    if timeclips is None:
        _, timeclips = clip_behavior_df(inter)
    if isinstance(timeclips, TimeClip):
        timeclips = TimeClipList(timeclips, labels="total")

    contact = pd.DataFrame()
    for label, timeclip in timeclips:
        clip_df, timeclip = clip_behavior_df(inter, timeclip)
        for k in range(1, inter.n_mice + 1):
            for j in range(1, inter.n_mice + 1):
                if k == j:
                    continue

                contact = pd.concat([contact, pd.DataFrame({
                    "id": k,
                    "id_other": j,
                    "duration": try_int(timeclips.duration()),
                    "bin": label,
                    "bin_width": try_int(timeclip.duration),
                    "bin_start": try_int(timeclip.start),
                    "bin_end": try_int(timeclip.end),

                    # マウスkがマウスjと接触した回数
                    "contact_count": np.sum(
                        clip_df[f"dist_{min(k, j)}_{max(k, j)}_event_type"]
                        == -1
                    ),
                    # マウスkがマウスjにアプローチした回数
                    "approach_count": np.sum(
                        (clip_df[f"dist_{min(k, j)}_{max(k, j)}_event_type"]
                         == -1)
                        & (clip_df[f"id_{k}_move_rolling_R"]
                           >= clip_df[f"id_{j}_move_rolling_R"])
                    ),
                    # マウスkがマウスjからアプローチされた回数
                    "receive_count": np.sum(
                        (clip_df[f"dist_{min(k, j)}_{max(k, j)}_event_type"]
                         == -1)
                        & (clip_df[f"id_{k}_move_rolling_R"]
                           < clip_df[f"id_{j}_move_rolling_R"])
                    ),

                    # マウスkがマウスjと接触した時間
                    "contact_duration": np.nansum(
                        (clip_df[f"dist_{min(k, j)}_{max(k, j)}_event_type"]
                         == -1)
                        * clip_df[
                            f"dist_{min(k, j)}_{max(k, j)}_event_duration"]
                    ),
                    # マウスkがマウスjにアプローチした時間
                    "approach_duration": np.nansum(
                        ((clip_df[f"dist_{min(k, j)}_{max(k, j)}_event_type"]
                          == -1)
                         & (clip_df[f"id_{k}_move_rolling_R"]
                            >= clip_df[f"id_{j}_move_rolling_R"]))
                        * clip_df[
                            f"dist_{min(k, j)}_{max(k, j)}_event_duration"]
                    ),
                    # マウスkがマウスjからアプローチされた時間
                    "receive_duration": np.nansum(
                        ((clip_df[f"dist_{min(k, j)}_{max(k, j)}_event_type"]
                          == -1)
                         & (clip_df[f"id_{k}_move_rolling_R"]
                            < clip_df[f"id_{j}_move_rolling_R"]))
                        * clip_df[
                            f"dist_{min(k, j)}_{max(k, j)}_event_duration"]
                    ),

                    "session": inter.session,
                    "camera": inter.camera,
                    "marker_id": inter.mice[k]["marker_id"],
                    "description": inter.mice[k]["description"]
                }, index=[k])])

    contact = contact.assign(
        # それぞれのidについてのinteractionの合計に占める、
        # 特定の相手id_otherの確率（回数または時間）を計算する。
        # _entropyという名前だが、この段階ではまだエントロピーではないので注意
        contact_count_entropy=lambda df:
            df["contact_count"]
            / df.groupby(["id", "bin"])["contact_count"].transform("sum"),
        approach_count_entropy=lambda df:
            df["approach_count"]
            / df.groupby(["id", "bin"])["approach_count"].transform("sum"),
        receive_count_entropy=lambda df:
            df["receive_count"]
            / df.groupby(["id", "bin"])["receive_count"].transform("sum"),
        contact_duration_entropy=lambda df:
            df["contact_duration"]
            / df.groupby(["id", "bin"])["contact_duration"].transform("sum"),
        approach_duration_entropy=lambda df:
            df["approach_duration"]
            / df.groupby(["id", "bin"])["approach_duration"].transform("sum"),
        receive_duration_entropy=lambda df:
            df["receive_duration"]
            / df.groupby(["id", "bin"])["receive_duration"].transform("sum")
    )

    # idとbinでgroup化して、それぞれのエントロピーを計算する
    timeclips_inter_entropy = contact.groupby(["id", "bin"]).agg({
        "duration": lambda s: np.unique(s)[0],
        "bin_width": lambda s: np.unique(s)[0],
        "bin_start": lambda s: np.unique(s)[0],
        "bin_end": lambda s: np.unique(s)[0],
        "contact_count_entropy": lambda s:
            -np.nansum(s.apply(lambda x: x * np.log2(x) if x != 0 else 0)),
        "approach_count_entropy": lambda s:
            -np.nansum(s.apply(lambda x: x * np.log2(x) if x != 0 else 0)),
        "receive_count_entropy": lambda s:
            -np.nansum(s.apply(lambda x: x * np.log2(x) if x != 0 else 0)),
        "contact_duration_entropy": lambda s:
            -np.nansum(s.apply(lambda x: x * np.log2(x) if x != 0 else 0)),
        "approach_duration_entropy": lambda s:
            -np.nansum(s.apply(lambda x: x * np.log2(x) if x != 0 else 0)),
        "receive_duration_entropy": lambda s:
            -np.nansum(s.apply(lambda x: x * np.log2(x) if x != 0 else 0)),
        "session": lambda s: np.unique(s)[0],
        "camera": lambda s: np.unique(s)[0],
        "marker_id": lambda s: np.unique(s)[0],
        "description": lambda s: np.unique(s)[0]
    }).reset_index()

    # 先頭から3列をid→duration→binの順番にする
    timeclips_inter_entropy = timeclips_inter_entropy. \
        set_index(["id", "duration", "bin", "bin_width", "bin_start"]). \
        sort_index(level=["bin_start", "bin", "id"]). \
        reset_index()

    return merge_project(timeclips_inter_entropy, proj)
