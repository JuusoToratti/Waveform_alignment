# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['Waveform_alingment.py'],
    pathex=[],
    binaries=[],
    datas=[('I2C_all.csv', '.'), ('REGRESSION_ AREC_X21_MADE2_I2C_1-I2C_0_P1V8(1) SDA-INC74024-2593-55585.csv', '.')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='Waveform_alingment',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
