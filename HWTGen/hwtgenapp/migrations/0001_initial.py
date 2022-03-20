# Generated by Django 3.1.7 on 2021-09-23 07:52

import django.core.validators
from django.db import migrations, models
import django.db.models.deletion
import phonenumber_field.modelfields


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='ContactInfo',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('username', models.CharField(blank=True, max_length=150, validators=[django.core.validators.MaxLengthValidator(150, message='Username cannot exceed 150 characters.')])),
                ('firstname', models.CharField(error_messages={'blank': 'First name is a required field.'}, max_length=150, validators=[django.core.validators.RegexValidator('^[A-Za-z ,.-]+$', 'First and last name cannot contain numbers.')])),
                ('lastname', models.CharField(error_messages={'blank': 'Last name is a required field.'}, max_length=150, validators=[django.core.validators.RegexValidator('^[A-Za-z ,.-]+$', 'First and last name cannot contain numbers.')])),
                ('email', models.EmailField(error_messages={'blank': 'Email is a required field.'}, max_length=254, null=True, validators=[django.core.validators.EmailValidator(code='invalid', message='Please enter a valid email address.')])),
                ('phonenumber', phonenumber_field.modelfields.PhoneNumberField(blank=True, max_length=50, region=None, validators=[django.core.validators.MaxLengthValidator(50, message='Phone number cannot exceed 50 characters.')])),
                ('message', models.TextField(max_length=1000, validators=[django.core.validators.MaxLengthValidator(1000, message='Message cannot exceed 1000 characters.')])),
                ('createdate', models.DateTimeField(blank=True, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='Topic',
            fields=[
                ('topicid', models.IntegerField(primary_key=True, serialize=False)),
                ('topicname', models.CharField(blank=True, max_length=50, null=True)),
                ('topicorderid', models.IntegerField(blank=True, null=True)),
                ('description', models.TextField(blank=True, null=True)),
                ('imagepath', models.CharField(blank=True, max_length=150, null=True)),
                ('url', models.CharField(blank=True, max_length=150, null=True)),
                ('createdby', models.CharField(blank=True, max_length=50, null=True)),
                ('createdate', models.DateTimeField(blank=True, null=True)),
                ('lastupdatedate', models.DateTimeField(blank=True, null=True)),
                ('sitemapenabled', models.BooleanField(default=False)),
                ('supertopicid', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='hwtgenapp.topic')),
            ],
        ),
        migrations.CreateModel(
            name='TopicDetail',
            fields=[
                ('topicdetailid', models.IntegerField(primary_key=True, serialize=False)),
                ('title', models.CharField(blank=True, max_length=50, null=True)),
                ('topicorderid', models.IntegerField(blank=True, null=True)),
                ('content', models.TextField(blank=True, null=True)),
                ('createdby', models.CharField(blank=True, max_length=50, null=True)),
                ('createdate', models.DateTimeField(blank=True, null=True)),
                ('lastupdatedate', models.DateTimeField(blank=True, null=True)),
                ('sitemapenabled', models.BooleanField(default=False)),
                ('topicid', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='hwtgenapp.topic')),
            ],
        ),
    ]
