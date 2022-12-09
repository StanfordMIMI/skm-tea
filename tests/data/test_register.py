from skm_tea.data.register import download_skm_tea_mini


def test_download_skm_tea_mini(tmpdir):
    """Test download skm_tea_mini."""
    output_dir = download_skm_tea_mini(track="dicom", download_path=tmpdir, force=True)
    assert str(output_dir) == str(tmpdir)
