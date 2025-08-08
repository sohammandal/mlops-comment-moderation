output "public_ip" {
  value = aws_instance.app.public_ip
}

output "public_dns" {
  value = aws_instance.app.public_dns
}

output "urls" {
  value = {
    api_docs  = "http://${aws_instance.app.public_dns}:8000/docs"
    streamlit = "http://${aws_instance.app.public_dns}:8501"
  }
}
